"""Perform ablation testing over k folds"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pmr.config import ProjectConfig
from pmr.paths import ProjectPaths
from pmr.prosody import EXTRACTORS
from pmr.train import fit, fold_split, load_examples, standardize_prosody

METRICS = ("rmse", "r2", "pcc", "ccc")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", *EXTRACTORS), default="base")
    parser.add_argument(
        "--processing-level", choices=("pair", "word"), default="pair"
    )
    parser.add_argument(
        "--supervision-level", choices=("track", "sentence"), default="track"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance**0.5


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = ProjectConfig()
    config = replace(
        config,
        model=replace(
            config.model,
            prosody_level=None if args.model == "base" else args.model,
            processing_level=args.processing_level,
            supervision_level=args.supervision_level,
        ),
        training=replace(
            config.training,
            **({"epochs": args.epochs} if args.epochs else {}),
        ),
    )
    config.validate()

    paths = ProjectPaths.discover()
    examples = load_examples(config, paths)
    if args.limit is not None:
        examples = examples[: args.limit]

    results = []
    for fold in range(args.folds):
        train_set, val_set, test_set = fold_split(examples, config, fold, args.folds)
        if config.model.prosody_level:
            train_set, val_set, test_set = standardize_prosody(
                train_set, val_set, test_set
            )
        name = f"ablation-{args.model}-{args.processing_level}-fold{fold}"
        _, test_metrics = fit(
            config, paths, train_set, val_set, test_set, name, config.seed + fold
        )
        results.append(test_metrics)

    for metric in METRICS:
        mean, std = mean_std([result[metric] for result in results])
        print(f"{metric}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
