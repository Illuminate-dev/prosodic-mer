"""Train the model"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pmr.config import ProjectConfig
from pmr.paths import ProjectPaths
from pmr.prosody import EXTRACTORS
from pmr.train import run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prosody-level", choices=tuple(EXTRACTORS), default=None)
    parser.add_argument(
        "--processing-level", choices=("pair", "word"), default="pair"
    )
    parser.add_argument(
        "--supervision-level", choices=("track", "sentence"), default="track"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = ProjectConfig()
    config = replace(
        config,
        model=replace(
            config.model,
            prosody_level=args.prosody_level,
            processing_level=args.processing_level,
            supervision_level=args.supervision_level,
        ),
    )
    config.validate()
    paths = ProjectPaths.discover()
    metrics = run(config, paths)
    print(
        "val: "
        + " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    )


if __name__ == "__main__":
    main()
