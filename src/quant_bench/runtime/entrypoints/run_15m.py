"""Traditional ML 15-minute runtime process."""

from quant_bench.runtime.entrypoints._shared import run


def main() -> None:
    run("traditional_15m")


if __name__ == "__main__":
    main()
