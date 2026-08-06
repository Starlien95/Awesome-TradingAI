# Portions adapted from Microsoft Qlib's scripts/dump_bin.py.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see LICENSES/QLIB-MIT.txt.
# Modifications copyright (c) 2026 quant-bench contributors.

from __future__ import annotations

import abc
import argparse
import logging
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Union
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

np = None
pd = None
tqdm = None
logger = logging.getLogger(__name__)
fname_to_code = None
code_to_fname = None


DEFAULT_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT", "ADA-USDT", "DOGE-USDT"]
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_INCLUDE_FIELDS = "open,high,low,close,volume,typical_price,vwap"
DEFAULT_MAX_RETRIES = 5
DEFAULT_REQUEST_TIMEOUT_MS = 20_000
TYPICAL_PRICE_SOURCE_FIELDS = ("high", "low", "close")

TIMEFRAME_TO_QLIB_FREQ = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
    "4h": "240min",
    "1d": "day",
}


def require_pandas():
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd
    return pd


def require_dump_dependencies() -> None:
    global np, tqdm, logger, fname_to_code, code_to_fname

    require_pandas()

    if np is None:
        import numpy as _np

        np = _np

    if tqdm is None:
        from tqdm import tqdm as _tqdm

        tqdm = _tqdm

    if not hasattr(logger, "success"):
        try:
            from loguru import logger as _logger

            logger = _logger
        except ModuleNotFoundError:
            logging.basicConfig(level=logging.INFO)

    if fname_to_code is None or code_to_fname is None:
        from qlib.utils import code_to_fname as _code_to_fname
        from qlib.utils import fname_to_code as _fname_to_code

        fname_to_code = _fname_to_code
        code_to_fname = _code_to_fname


def get_qlib_freq(timeframe: str) -> str:
    return TIMEFRAME_TO_QLIB_FREQ.get(timeframe, timeframe)


def default_raw_csv_path(timeframe: str) -> str:
    return f"./csv_data/all_crypto_data_{timeframe}.csv"


def default_split_dir(timeframe: str) -> str:
    return f"./split_csvs/{timeframe}"


def default_qlib_dir(timeframe: str) -> str:
    return f"./qlib_data/{timeframe}"


def parse_symbols(symbols: str) -> list[str]:
    return [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]


def fill_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add a documented typical-price proxy for Qlib's required vwap field.

    OHLCV bars do not contain the trade-level price and quote-volume data needed
    to calculate true VWAP. The compatibility ``vwap`` column therefore mirrors
    ``typical_price`` and carries an explicit ``vwap_method`` marker.
    """
    require_pandas()
    missing_fields = [field for field in TYPICAL_PRICE_SOURCE_FIELDS if field not in df.columns]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Cannot calculate the typical-price proxy; missing fields: {missing}")

    result = df.copy()
    prices = result.loc[:, list(TYPICAL_PRICE_SOURCE_FIELDS)].apply(pd.to_numeric, errors="coerce")
    result["typical_price"] = (prices["high"] + prices["low"] + prices["close"]) / 3.0
    result["vwap"] = result["typical_price"]
    result["vwap_method"] = "typical_price_proxy"
    return result


def download_okx_data_with_ccxt(
    symbol: str,
    timeframe: str,
    start_date: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> pd.DataFrame:
    require_pandas()
    import ccxt

    print(f"【CCXT】正在下载 {symbol} [{timeframe}]，起始: {start_date}...")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    exchange = ccxt.okx({"enableRateLimit": True, "timeout": DEFAULT_REQUEST_TIMEOUT_MS})
    ccxt_symbol = symbol.replace("-", "/")
    ccxt_timeframe = timeframe.lower()

    try:
        since_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    except ValueError:
        print(f"日期格式错误: {start_date}，请使用 YYYY-MM-DD")
        return pd.DataFrame()

    all_klines = []
    consecutive_failures = 0
    while True:
        try:
            klines = exchange.fetch_ohlcv(ccxt_symbol, ccxt_timeframe, since=since_timestamp, limit=100)
            consecutive_failures = 0
            if not klines:
                print(f"\n  -> {symbol} 下载完毕。")
                break

            oldest_ts, latest_ts = klines[0][0], klines[-1][0]
            print(
                f"\r  -> 获取: {datetime.fromtimestamp(oldest_ts / 1000)}"
                f" -> {datetime.fromtimestamp(latest_ts / 1000)}",
                end="",
            )
            all_klines.extend(klines)

            duration_ms = exchange.parse_timeframe(ccxt_timeframe) * 1000
            since_timestamp = latest_ts + duration_ms
            time.sleep(0.05)
        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            print(f"\n  请求异常: {exc}")
            if consecutive_failures > max_retries:
                raise RuntimeError(
                    f"CCXT download failed after {max_retries} retries for {symbol} {timeframe}"
                ) from exc
            time.sleep(min(2**consecutive_failures, 30))

    if not all_klines:
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["instrument"] = symbol

    qlib_df = df[["datetime", "instrument", "open", "high", "low", "close", "volume"]]
    qlib_df = (
        qlib_df.sort_values("datetime")
        .drop_duplicates(subset=["datetime", "instrument"], keep="first")
        .reset_index(drop=True)
    )
    qlib_df = fill_vwap(qlib_df)

    print(f"  -> 有效数据: {len(qlib_df)} 条")
    return qlib_df


def save_raw_csv(all_symbols_df: pd.DataFrame, csv_path: str, timeframe: str | None = None) -> None:
    require_pandas()
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在保存至: {target}...")
    output_df = all_symbols_df.rename(columns={"datetime": "timestamp", "instrument": "symbol"})
    output_df["timestamp"] = pd.to_datetime(output_df["timestamp"], utc=True)
    output_df["date"] = output_df["timestamp"]
    output_df["source"] = "okx_ccxt"
    output_df["frequency"] = timeframe or "unknown"
    output_df = fill_vwap(output_df)
    output_fields = [
        "timestamp",
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "typical_price",
        "vwap",
        "vwap_method",
        "source",
        "frequency",
    ]
    output_df.to_csv(target, columns=output_fields, index=False)
    print("保存完成。")


def run_download(timeframe: str, start: str, symbols: str, output: str | None = None) -> str | None:
    require_pandas()
    output = output or default_raw_csv_path(timeframe)
    selected_symbols = parse_symbols(symbols)

    print("=" * 50)
    print("下载原始 K 线数据")
    print(f"时间周期: {timeframe}")
    print(f"开始时间: {start}")
    print(f"目标文件: {output}")
    print("=" * 50)

    all_dfs = []
    for symbol in selected_symbols:
        df = download_okx_data_with_ccxt(symbol, timeframe, start)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("未能下载到任何数据，请检查网络或配置。")
        return None

    master_df = pd.concat(all_dfs).reset_index(drop=True)
    print("-" * 30)
    print(f"合并完成，总数据量: {len(master_df)} 条")
    save_raw_csv(master_df, output, timeframe)
    return output


def run_split(timeframe: str, source_file: str | None = None, output_dir: str | None = None) -> str:
    require_pandas()
    source_csv_path = source_file or default_raw_csv_path(timeframe)
    output_dir = output_dir or default_split_dir(timeframe)

    if not Path(source_csv_path).exists():
        raise FileNotFoundError(f"找不到源文件: {source_csv_path}")

    print(f"正在读取原始数据: {source_csv_path}")
    df = pd.read_csv(source_csv_path)

    if "symbol" not in df.columns:
        raise ValueError("CSV 文件中缺少 'symbol' 列，无法拆分。")
    df = fill_vwap(df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"拆分输出目录: {output_path.resolve()}")

    count = 0
    for symbol, group_df in df.groupby("symbol"):
        safe_symbol_name = symbol.replace("/", "_")
        group_df.to_csv(output_path / f"{safe_symbol_name}.csv", index=False)
        count += 1

    print(f"拆分完成，共处理 {count} 个交易对。")
    return output_dir


def read_as_df(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    require_pandas()
    """
    Read a csv or parquet file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the data file.
    **kwargs :
        Additional keyword arguments passed to the underlying pandas
        reader.

    Returns
    -------
    pd.DataFrame
    """
    file_path = Path(file_path).expanduser()
    suffix = file_path.suffix.lower()

    keep_keys = {".csv": ("low_memory",)}
    kept_kwargs = {}
    for k in keep_keys.get(suffix, []):
        if k in kwargs:
            kept_kwargs[k] = kwargs[k]

    if suffix == ".csv":
        return pd.read_csv(file_path, **kept_kwargs)
    elif suffix == ".parquet":
        return pd.read_parquet(file_path, **kept_kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


class DumpDataBase:
    INSTRUMENTS_START_FIELD = "start_datetime"
    INSTRUMENTS_END_FIELD = "end_datetime"
    CALENDARS_DIR_NAME = "calendars"
    FEATURES_DIR_NAME = "features"
    INSTRUMENTS_DIR_NAME = "instruments"
    DUMP_FILE_SUFFIX = ".bin"
    DAILY_FORMAT = "%Y-%m-%d"
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"
    INSTRUMENTS_SEP = "\t"
    INSTRUMENTS_FILE_NAME = "all.txt"

    UPDATE_MODE = "update"
    ALL_MODE = "all"

    def __init__(
        self,
        data_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        data_path: str
            stock data path or directory
        qlib_dir: str
            qlib(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup qlib_dir to backup_dir
        freq: str, default "day"
            transaction frequency
        max_workers: int, default None
            number of threads
        date_field_name: str, default "date"
            the name of the date field in the csv
        file_suffix: str, default ".csv"
            file suffix
        symbol_field_name: str, default "symbol"
            symbol field name
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        """
        data_path = Path(data_path).expanduser()
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        self.file_suffix = file_suffix
        self.symbol_field_name = symbol_field_name
        self.df_files = sorted(data_path.glob(f"*{self.file_suffix}") if data_path.is_dir() else [data_path])
        if limit_nums is not None:
            self.df_files = self.df_files[: int(limit_nums)]
        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        if backup_dir is not None:
            self._backup_qlib_dir(Path(backup_dir).expanduser())

        self.freq = freq
        self.calendar_format = self.DAILY_FORMAT if self.freq == "day" else self.HIGH_FREQ_FORMAT

        self.works = max_workers
        self.date_field_name = date_field_name

        self._calendars_dir = self.qlib_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.qlib_dir.joinpath(self.FEATURES_DIR_NAME)
        self._instruments_dir = self.qlib_dir.joinpath(self.INSTRUMENTS_DIR_NAME)

        self._calendars_list = []

        self._mode = self.ALL_MODE
        self._kwargs = {}

    def _backup_qlib_dir(self, target_dir: Path):
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    def _format_datetime(self, datetime_d: [str, pd.Timestamp]):
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.calendar_format)

    def _get_date(
        self, file_or_df: [Path, pd.DataFrame], *, is_begin_end: bool = False, as_set: bool = False
    ) -> Iterable[pd.Timestamp]:
        if not isinstance(file_or_df, pd.DataFrame):
            df = self._get_source_data(file_or_df)
        else:
            df = file_or_df
        if df.empty or self.date_field_name not in df.columns.tolist():
            _calendars = pd.Series(dtype=np.float32)
        else:
            _calendars = df[self.date_field_name]

        if is_begin_end and as_set:
            return (_calendars.min(), _calendars.max()), set(_calendars)
        elif is_begin_end:
            return _calendars.min(), _calendars.max()
        elif as_set:
            return set(_calendars)
        else:
            return _calendars.tolist()

    def _get_source_data(self, file_path: Path) -> pd.DataFrame:
        df = read_as_df(file_path, low_memory=False)
        df = fill_vwap(df)
        if self.date_field_name in df.columns:
            parsed = pd.to_datetime(df[self.date_field_name], errors="raise", utc=True)
            df[self.date_field_name] = parsed.dt.tz_convert(None)
        # df.drop_duplicates([self.date_field_name], inplace=True)
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        return fname_to_code(file_path.stem.strip().lower())

    def get_dump_fields(self, df_columns: Iterable[str]) -> Iterable[str]:
        return (
            self._include_fields
            if self._include_fields
            else set(df_columns) - set(self._exclude_fields) if self._exclude_fields else df_columns
        )

    @staticmethod
    def _read_calendars(calendar_path: Path) -> List[pd.Timestamp]:
        return sorted(
            map(
                pd.Timestamp,
                pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
            )
        )

    def _read_instruments(self, instrument_path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            instrument_path,
            sep=self.INSTRUMENTS_SEP,
            names=[
                self.symbol_field_name,
                self.INSTRUMENTS_START_FIELD,
                self.INSTRUMENTS_END_FIELD,
            ],
        )

        return df

    def save_calendars(self, calendars_data: list):
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        calendars_path = str(self._calendars_dir.joinpath(f"{self.freq}.txt").expanduser().resolve())
        result_calendars_list = [self._format_datetime(x) for x in calendars_data]
        np.savetxt(calendars_path, result_calendars_list, fmt="%s", encoding="utf-8")

    def save_instruments(self, instruments_data: Union[list, pd.DataFrame]):
        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        instruments_path = str(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME).resolve())
        if isinstance(instruments_data, pd.DataFrame):
            _df_fields = [self.symbol_field_name, self.INSTRUMENTS_START_FIELD, self.INSTRUMENTS_END_FIELD]
            instruments_data = instruments_data.loc[:, _df_fields]
            instruments_data[self.symbol_field_name] = instruments_data[self.symbol_field_name].apply(
                lambda x: fname_to_code(x.lower()).upper()
            )
            instruments_data.to_csv(instruments_path, header=False, sep=self.INSTRUMENTS_SEP, index=False)
        else:
            np.savetxt(instruments_path, instruments_data, fmt="%s", encoding="utf-8")

    def data_merge_calendar(self, df: pd.DataFrame, calendars_list: List[pd.Timestamp]) -> pd.DataFrame:
        # calendars
        calendars_df = pd.DataFrame(data=calendars_list, columns=[self.date_field_name])
        calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype("datetime64[ns]")
        cal_df = calendars_df[
            (calendars_df[self.date_field_name] >= df[self.date_field_name].min())
            & (calendars_df[self.date_field_name] <= df[self.date_field_name].max())
        ]
        # align index
        cal_df.set_index(self.date_field_name, inplace=True)
        df.set_index(self.date_field_name, inplace=True)
        r_df = df.reindex(cal_df.index)
        return r_df

    @staticmethod
    def get_datetime_index(df: pd.DataFrame, calendar_list: List[pd.Timestamp]) -> int:
        return calendar_list.index(df.index.min())

    def _data_to_bin(self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path):
        if df.empty:
            logger.warning(f"{features_dir.name} data is None or empty")
            return
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        # align index
        _df = self.data_merge_calendar(df, calendar_list)
        if _df.empty:
            logger.warning(f"{features_dir.name} data is not in calendars")
            return
        # used when creating a bin file
        date_index = self.get_datetime_index(_df, calendar_list)
        for field in self.get_dump_fields(_df.columns):
            bin_path = features_dir.joinpath(f"{field.lower()}.{self.freq}{self.DUMP_FILE_SUFFIX}")
            if field not in _df.columns:
                continue
            if bin_path.exists() and self._mode == self.UPDATE_MODE:
                # update
                with bin_path.open("ab") as fp:
                    np.array(_df[field]).astype("<f").tofile(fp)
            else:
                # append; self._mode == self.ALL_MODE or not bin_path.exists()
                np.hstack([date_index, _df[field]]).astype("<f").tofile(str(bin_path.resolve()))

    def _dump_bin(self, file_or_data: [Path, pd.DataFrame], calendar_list: List[pd.Timestamp]):
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        if isinstance(file_or_data, pd.DataFrame):
            if file_or_data.empty:
                return
            code = fname_to_code(str(file_or_data.iloc[0][self.symbol_field_name]).lower())
            df = file_or_data
        elif isinstance(file_or_data, Path):
            code = self.get_symbol_from_file(file_or_data)
            df = self._get_source_data(file_or_data)
        else:
            raise ValueError(f"not support {type(file_or_data)}")
        if df is None or df.empty:
            logger.warning(f"{code} data is None or empty")
            return

        # try to remove dup rows or it will cause exception when reindex.
        df = df.drop_duplicates(self.date_field_name)

        # features save dir
        features_dir = self._features_dir.joinpath(code_to_fname(code).lower())
        features_dir.mkdir(parents=True, exist_ok=True)
        self._data_to_bin(df, calendar_list, features_dir)

    @abc.abstractmethod
    def dump(self):
        raise NotImplementedError("dump not implemented!")

    def __call__(self, *args, **kwargs):
        self.dump()


class DumpDataAll(DumpDataBase):
    def _get_all_date(self):
        logger.info("start get all date......")
        all_datetime = set()
        date_range_list = []
        _fun = partial(self._get_date, as_set=True, is_begin_end=True)
        with tqdm(total=len(self.df_files)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as executor:
                for file_path, ((_begin_time, _end_time), _set_calendars) in zip(
                    self.df_files, executor.map(_fun, self.df_files)
                ):
                    all_datetime = all_datetime | _set_calendars
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(_end_time, pd.Timestamp):
                        _begin_time = self._format_datetime(_begin_time)
                        _end_time = self._format_datetime(_end_time)
                        symbol = self.get_symbol_from_file(file_path)
                        _inst_fields = [symbol.upper(), _begin_time, _end_time]
                        date_range_list.append(f"{self.INSTRUMENTS_SEP.join(_inst_fields)}")
                    p_bar.update()
        self._kwargs["all_datetime_set"] = all_datetime
        self._kwargs["date_range_list"] = date_range_list
        logger.info("end of get all date.\n")

    def _dump_calendars(self):
        logger.info("start dump calendars......")
        self._calendars_list = sorted(map(pd.Timestamp, self._kwargs["all_datetime_set"]))
        self.save_calendars(self._calendars_list)
        logger.info("end of calendars dump.\n")

    def _dump_instruments(self):
        logger.info("start dump instruments......")
        self.save_instruments(self._kwargs["date_range_list"])
        logger.info("end of instruments dump.\n")

    def _dump_features(self):
        logger.info("start dump features......")
        _dump_func = partial(self._dump_bin, calendar_list=self._calendars_list)
        with tqdm(total=len(self.df_files)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, self.df_files):
                    p_bar.update()

        logger.info("end of features dump.\n")

    def dump(self):
        self._get_all_date()
        self._dump_calendars()
        self._dump_instruments()
        self._dump_features()


class DumpDataFix(DumpDataAll):
    def _dump_instruments(self):
        logger.info("start dump instruments......")
        _fun = partial(self._get_date, is_begin_end=True)
        new_stock_files = sorted(
            filter(
                lambda x: self.get_symbol_from_file(x).upper() not in self._old_instruments,
                self.df_files,
            )
        )
        with tqdm(total=len(new_stock_files)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as execute:
                for file_path, (_begin_time, _end_time) in zip(new_stock_files, execute.map(_fun, new_stock_files)):
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(_end_time, pd.Timestamp):
                        symbol = self.get_symbol_from_file(file_path).upper()
                        _dt_map = self._old_instruments.setdefault(symbol, dict())
                        _dt_map[self.INSTRUMENTS_START_FIELD] = self._format_datetime(_begin_time)
                        _dt_map[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end_time)
                    p_bar.update()
        _inst_df = pd.DataFrame.from_dict(self._old_instruments, orient="index")
        _inst_df.index.names = [self.symbol_field_name]
        self.save_instruments(_inst_df.reset_index())
        logger.info("end of instruments dump.\n")

    def dump(self):
        self._calendars_list = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))
        # noinspection PyAttributeOutsideInit
        self._old_instruments = (
            self._read_instruments(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME))
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )  # type: dict
        self._dump_instruments()
        self._dump_features()


class DumpDataUpdate(DumpDataBase):
    def __init__(
        self,
        data_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        data_path: str
            stock data path or directory
        qlib_dir: str
            qlib(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup qlib_dir to backup_dir
        freq: str, default "day"
            transaction frequency
        max_workers: int, default None
            number of threads
        date_field_name: str, default "date"
            the name of the date field in the csv
        file_suffix: str, default ".csv"
            file suffix
        symbol_field_name: str, default "symbol"
            symbol field name
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        """
        super().__init__(
            data_path,
            qlib_dir,
            backup_dir,
            freq,
            max_workers,
            date_field_name,
            file_suffix,
            symbol_field_name,
            exclude_fields,
            include_fields,
            limit_nums,
        )
        self._mode = self.UPDATE_MODE
        self._old_calendar_list = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))
        # NOTE: all.txt only exists once for each stock
        # NOTE: if a stock corresponds to multiple different time ranges, user need to modify self._update_instruments
        self._update_instruments = (
            self._read_instruments(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME))
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )  # type: dict

        # load all csv files
        self._all_data = self._load_all_source_data()  # type: pd.DataFrame
        self._new_calendar_list = self._old_calendar_list + sorted(
            filter(lambda x: x > self._old_calendar_list[-1], self._all_data[self.date_field_name].unique())
        )

    def _load_all_source_data(self):
        # NOTE: Need more memory
        logger.info("start load all source data....")
        all_df = []

        def _read_df(file_path: Path):
            _df = read_as_df(file_path)
            if self.date_field_name in _df.columns and not np.issubdtype(
                _df[self.date_field_name].dtype, np.datetime64
            ):
                _df[self.date_field_name] = pd.to_datetime(_df[self.date_field_name])
            if self.symbol_field_name not in _df.columns:
                _df[self.symbol_field_name] = self.get_symbol_from_file(file_path)
            return _df

        with tqdm(total=len(self.df_files)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as executor:
                for df in executor.map(_read_df, self.df_files):
                    if not df.empty:
                        all_df.append(df)
                    p_bar.update()

        logger.info("end of load all data.\n")
        return pd.concat(all_df, sort=False)

    def _dump_calendars(self):
        pass

    def _dump_instruments(self):
        pass

    def _dump_features(self):
        logger.info("start dump features......")
        error_code = {}
        with ThreadPoolExecutor(max_workers=self.works) as executor:
            futures = {}
            for _code, _df in self._all_data.groupby(self.symbol_field_name, group_keys=False):
                _code = fname_to_code(str(_code).lower()).upper()
                _start, _end = self._get_date(_df, is_begin_end=True)
                if not (isinstance(_start, pd.Timestamp) and isinstance(_end, pd.Timestamp)):
                    continue
                if _code in self._update_instruments:
                    # exists stock, will append data
                    _update_calendars = (
                        _df[_df[self.date_field_name] > self._update_instruments[_code][self.INSTRUMENTS_END_FIELD]][
                            self.date_field_name
                        ]
                        .sort_values()
                        .to_list()
                    )
                    if _update_calendars:
                        self._update_instruments[_code][self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                        futures[executor.submit(self._dump_bin, _df, _update_calendars)] = _code
                else:
                    # new stock
                    _dt_range = self._update_instruments.setdefault(_code, dict())
                    _dt_range[self.INSTRUMENTS_START_FIELD] = self._format_datetime(_start)
                    _dt_range[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                    futures[executor.submit(self._dump_bin, _df, self._new_calendar_list)] = _code

            with tqdm(total=len(futures)) as p_bar:
                for _future in as_completed(futures):
                    try:
                        _future.result()
                    except Exception:
                        error_code[futures[_future]] = traceback.format_exc()
                    p_bar.update()
            logger.info(f"dump bin errors: {error_code}")

        logger.info("end of features dump.\n")

    def dump(self):
        self.save_calendars(self._new_calendar_list)
        self._dump_features()
        df = pd.DataFrame.from_dict(self._update_instruments, orient="index")
        df.index.names = [self.symbol_field_name]
        self.save_instruments(df.reset_index())


def run_dump(
    mode: str,
    timeframe: str,
    data_path: str | None = None,
    qlib_dir: str | None = None,
    backup_dir: str | None = None,
    freq: str | None = None,
    max_workers: int = 16,
    date_field_name: str = "auto",
    file_suffix: str = ".csv",
    symbol_field_name: str = "symbol",
    exclude_fields: str = "",
    include_fields: str = DEFAULT_INCLUDE_FIELDS,
    limit_nums: int | None = None,
) -> dict[str, Any]:
    require_dump_dependencies()

    data_path = data_path or default_split_dir(timeframe)
    qlib_dir = qlib_dir or default_qlib_dir(timeframe)
    freq = freq or get_qlib_freq(timeframe)

    source_path = Path(data_path).expanduser()
    source_files = sorted(
        source_path.glob(f"*{file_suffix}") if source_path.is_dir() else [source_path]
    )
    if limit_nums is not None:
        source_files = source_files[: int(limit_nums)]
    if not source_files or any(not path.is_file() for path in source_files):
        raise FileNotFoundError(f"no Qlib source CSV files found: {source_path}")
    headers = {path: set(pd.read_csv(path, nrows=0).columns) for path in source_files}
    requested_date_field = str(date_field_name or "auto").strip()
    if requested_date_field == "auto":
        if all("date" in columns for columns in headers.values()):
            date_field_name = "date"
        elif all("timestamp" in columns for columns in headers.values()):
            date_field_name = "timestamp"
        else:
            details = ", ".join(
                f"{path.name}={sorted(columns)}" for path, columns in headers.items()
            )
            raise ValueError(
                "Qlib source files require one common date or timestamp column; " + details
            )
    else:
        date_field_name = requested_date_field
        missing = [path.name for path, columns in headers.items() if date_field_name not in columns]
        if missing:
            raise ValueError(
                f"Qlib date field {date_field_name!r} is missing from source files: {missing}"
            )
    empty = [
        path.name
        for path in source_files
        if pd.read_csv(path, usecols=[date_field_name], nrows=1).empty
    ]
    if empty:
        raise ValueError(f"Qlib source CSV files contain no data rows: {empty}")

    dump_cls = {
        "dump_all": DumpDataAll,
        "dump_fix": DumpDataFix,
        "dump_update": DumpDataUpdate,
    }[mode]

    dumper = dump_cls(
        data_path=data_path,
        qlib_dir=qlib_dir,
        backup_dir=backup_dir,
        freq=freq,
        max_workers=max_workers,
        date_field_name=date_field_name,
        file_suffix=file_suffix,
        symbol_field_name=symbol_field_name,
        exclude_fields=exclude_fields,
        include_fields=include_fields,
        limit_nums=limit_nums,
    )
    dumper.dump()
    output_root = Path(qlib_dir).expanduser().resolve()
    calendar_path = output_root / "calendars" / f"{freq}.txt"
    instruments_path = output_root / "instruments" / "all.txt"
    feature_files = [
        path for path in (output_root / "features").rglob("*.bin") if path.stat().st_size > 4
    ] if (output_root / "features").is_dir() else []
    if not calendar_path.is_file() or not calendar_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"Qlib dump produced an empty calendar: {calendar_path}")
    if not instruments_path.is_file() or not instruments_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"Qlib dump produced an empty instrument registry: {instruments_path}")
    if not feature_files:
        raise RuntimeError(f"Qlib dump produced no non-empty feature bins: {output_root / 'features'}")
    return {
        "qlib_dir": str(output_root),
        "mode": mode,
        "frequency": freq,
        "date_field_name": date_field_name,
        "source_files": len(source_files),
        "calendar_rows": sum(1 for line in calendar_path.read_text(encoding="utf-8").splitlines() if line),
        "instruments": sum(1 for line in instruments_path.read_text(encoding="utf-8").splitlines() if line),
        "feature_bins": len(feature_files),
    }


def run_pipeline(args: argparse.Namespace) -> None:
    raw_csv = run_download(args.timeframe, args.start, args.symbols, args.output)
    if raw_csv is None:
        return
    split_dir = run_split(args.timeframe, raw_csv, args.split_dir)
    run_dump(
        mode="dump_all",
        timeframe=args.timeframe,
        data_path=split_dir,
        qlib_dir=args.qlib_dir,
        freq=args.freq,
        max_workers=args.max_workers,
        include_fields=args.include_fields,
    )


def add_timeframe_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeframe",
        "-t",
        default="1m",
        help="K 线周期，例如 1m, 5m, 15m, 1h, 4h, 1d。",
    )


def add_dump_args(parser: argparse.ArgumentParser) -> None:
    add_timeframe_arg(parser)
    parser.add_argument("--data-path", "--data_path", default=None, help="逐 symbol CSV 目录。")
    parser.add_argument("--qlib-dir", "--qlib_dir", default=None, help="Qlib bin 输出目录。")
    parser.add_argument("--backup-dir", "--backup_dir", default=None, help="更新前备份目录。")
    parser.add_argument("--freq", default=None, help="Qlib 频率，例如 1min, 5min, 60min, day。")
    parser.add_argument("--max-workers", "--max_workers", type=int, default=16)
    parser.add_argument("--date-field-name", "--date_field_name", default="date")
    parser.add_argument("--file-suffix", "--file_suffix", default=".csv")
    parser.add_argument("--symbol-field-name", "--symbol_field_name", default="symbol")
    parser.add_argument("--exclude-fields", "--exclude_fields", default="")
    parser.add_argument("--include-fields", "--include_fields", default=DEFAULT_INCLUDE_FIELDS)
    parser.add_argument("--limit-nums", "--limit_nums", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qlib crypto data preparation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="下载 OKX K 线数据为总 CSV。")
    add_timeframe_arg(download_parser)
    download_parser.add_argument("--start", "-s", default=DEFAULT_START_DATE, help="开始日期 YYYY-MM-DD。")
    download_parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="逗号分隔交易对列表，例如 BTC-USDT,ETH-USDT。",
    )
    download_parser.add_argument("--output", "-o", default=None, help="输出总 CSV 路径。")

    split_parser = subparsers.add_parser("split", help="按 symbol 拆分总 CSV。")
    add_timeframe_arg(split_parser)
    split_parser.add_argument("--file", "-f", default=None, help="源 CSV 路径。")
    split_parser.add_argument("--output-dir", "--output_dir", default=None, help="拆分输出目录。")

    for command in ("dump_all", "dump_fix", "dump_update"):
        dump_parser = subparsers.add_parser(command, help=f"执行 {command} Qlib bin 转换。")
        add_dump_args(dump_parser)

    pipeline_parser = subparsers.add_parser("pipeline", help="依次执行 download、split、dump_all。")
    add_timeframe_arg(pipeline_parser)
    pipeline_parser.add_argument("--start", "-s", default=DEFAULT_START_DATE, help="开始日期 YYYY-MM-DD。")
    pipeline_parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="逗号分隔交易对列表，例如 BTC-USDT,ETH-USDT。",
    )
    pipeline_parser.add_argument("--output", "-o", default=None, help="下载总 CSV 路径。")
    pipeline_parser.add_argument("--split-dir", "--split_dir", default=None, help="拆分输出目录。")
    pipeline_parser.add_argument("--qlib-dir", "--qlib_dir", default=None, help="Qlib bin 输出目录。")
    pipeline_parser.add_argument("--freq", default=None, help="Qlib 频率，例如 1min, 5min, 60min, day。")
    pipeline_parser.add_argument("--max-workers", "--max_workers", type=int, default=16)
    pipeline_parser.add_argument("--include-fields", "--include_fields", default=DEFAULT_INCLUDE_FIELDS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        run_download(args.timeframe, args.start, args.symbols, args.output)
    elif args.command == "split":
        run_split(args.timeframe, args.file, args.output_dir)
    elif args.command in {"dump_all", "dump_fix", "dump_update"}:
        run_dump(
            mode=args.command,
            timeframe=args.timeframe,
            data_path=args.data_path,
            qlib_dir=args.qlib_dir,
            backup_dir=args.backup_dir,
            freq=args.freq,
            max_workers=args.max_workers,
            date_field_name=args.date_field_name,
            file_suffix=args.file_suffix,
            symbol_field_name=args.symbol_field_name,
            exclude_fields=args.exclude_fields,
            include_fields=args.include_fields,
            limit_nums=args.limit_nums,
        )
    elif args.command == "pipeline":
        run_pipeline(args)


if __name__ == "__main__":
    main()
