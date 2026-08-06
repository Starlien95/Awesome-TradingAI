"""MacroHFT v1 4-hour runtime process."""

from quant_bench.runtime.entrypoints._shared import run


def main() -> None:
    run("macrohft_4h")


if __name__ == "__main__":
    main()
