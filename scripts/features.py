"""Extract VGGish, ALBERT, and prosody features"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pmr.config import ProjectConfig
from pmr.features import feature_dataset
from pmr.paths import ProjectPaths
from pmr.prosody import prosody_dataset

MODALITIES = ("audio", "text", "prosody")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--track-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--modality",
        nargs="+",
        choices=MODALITIES,
        default=("audio", "text"),
        help="modalities to extract (default: audio text)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = ProjectConfig()
    config.validate()
    paths = ProjectPaths.discover()
    modalities = set(args.modality)
    track_ids: list[str] = []
    if modalities & {"audio", "text"}:
        track_ids += feature_dataset(
            config,
            paths,
            track_ids=args.track_ids,
            limit=args.limit,
            overwrite=args.overwrite,
            modalities=modalities,
        )
    if "prosody" in modalities:
        track_ids += prosody_dataset(
            config,
            paths,
            track_ids=args.track_ids,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    if track_ids:
        print(f"extracted features for {len(set(track_ids))} track(s)")


if __name__ == "__main__":
    main()
