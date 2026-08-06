"""MacroHFT v1 15-minute runtime process."""

from quant_bench.runtime.entrypoints._shared import run


def main() -> None:
    run("macrohft_15m")


if __name__ == "__main__":
    main()
