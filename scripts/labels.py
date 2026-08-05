"""Build valence-arousal labels from DEAM"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pmr.config import ProjectConfig
from pmr.labels import label_dataset
from pmr.paths import ProjectPaths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--track-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = ProjectConfig()
    config.validate()
    paths = ProjectPaths.discover()
    results = label_dataset(
        config,
        paths,
        track_ids=args.track_ids,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    if results:
        print(f"labelled {len(results)} track(s)")


if __name__ == "__main__":
    main()
