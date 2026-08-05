"""Download and extract the DEAM dataset"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pmr.config import ProjectConfig
from pmr.deam import fetch_deam
from pmr.paths import ProjectPaths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = ProjectConfig()
    config.validate()
    paths = ProjectPaths.discover()
    count = fetch_deam(config, paths)
    print(f"fetched {count} track(s)")


if __name__ == "__main__":
    main()
