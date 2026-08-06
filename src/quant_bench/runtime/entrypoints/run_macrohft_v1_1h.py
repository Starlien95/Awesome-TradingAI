"""MacroHFT v1 1-hour runtime process."""

from quant_bench.runtime.entrypoints._shared import run


def main() -> None:
    run("macrohft_1h")


if __name__ == "__main__":
    main()
