import json
import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from pmr.config import ProjectConfig
from pmr.metrics import evaluate
from pmr.model import build_emotion_regressor
from pmr.paths import ProjectPaths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Example:
    track_id: str
    melody: torch.Tensor
    lyric: torch.Tensor
    chorus: torch.Tensor
    target: torch.Tensor


def load_feature(
    paths: ProjectPaths, dataset: str, unit: str, modality: str, track_id: str
):
    path = paths.features(dataset, unit, modality) / f"{track_id}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["features"], data["start"], data["end"]


def load_chorus(
    paths: ProjectPaths,
    dataset: str,
    track_id: str,
    starts: np.ndarray,
    ends: np.ndarray,
) -> torch.Tensor:
    chorus = np.zeros(len(starts), dtype=bool)
    path = paths.structures(dataset) / f"{track_id}.json"
    if path.exists():
        sections = json.loads(path.read_text())["sections"]
        for index, (start, end) in enumerate(zip(starts, ends)):
            midpoint = (start + end) / 2
            for section in sections:
                if section["start"] <= midpoint <= section["end"]:
                    chorus[index] = section["label"] == "chorus"
                    break
    return torch.tensor(chorus)


def load_examples(config: ProjectConfig, paths: ProjectPaths) -> list[Example]:
    dataset = config.data.dataset
    unit = config.data.features.unit
    label_dir = paths.labels(dataset)
    label_paths = sorted(label_dir.glob("*.npz")) if label_dir.exists() else []

    examples: list[Example] = []
    skipped = 0
    for label_path in label_paths:
        track_id = label_path.stem
        melody = load_feature(paths, dataset, unit, "vggish", track_id)
        lyric = load_feature(paths, dataset, unit, "albert", track_id)
        values = np.load(label_path)["values"]
        if melody is None or lyric is None:
            logger.warning("skipping %s (missing features)", track_id)
            skipped += 1
            continue
        features, starts, ends = melody
        if len(values) != len(features) or len(lyric[0]) != len(features):
            logger.warning("skipping %s (length mismatch)", track_id)
            skipped += 1
            continue
        target = np.nanmean(values, axis=0)
        if np.isnan(target).any():
            logger.warning("skipping %s (no annotated units)", track_id)
            skipped += 1
            continue
        examples.append(
            Example(
                track_id=track_id,
                melody=torch.tensor(features, dtype=torch.float32),
                lyric=torch.tensor(lyric[0], dtype=torch.float32),
                chorus=load_chorus(paths, dataset, track_id, starts, ends),
                target=torch.tensor(target, dtype=torch.float32),
            )
        )
    logger.info("loaded %d track(s) (%d skipped)", len(examples), skipped)
    return examples


def split_examples(
    examples: list[Example], config: ProjectConfig
) -> tuple[list[Example], list[Example], list[Example]]:
    by_id = {example.track_id: example for example in examples}
    ids = sorted(by_id)
    random.Random(config.seed).shuffle(ids)
    count = len(ids)
    test = int(count * config.training.test_fraction)
    val = int(count * config.training.val_fraction)
    return (
        [by_id[i] for i in ids[test + val :]],
        [by_id[i] for i in ids[test : test + val]],
        [by_id[i] for i in ids[:test]],
    )


def collate(examples: list[Example]):
    length = max(len(example.melody) for example in examples)
    batch = len(examples)
    melody = torch.zeros(batch, length, examples[0].melody.shape[-1])
    lyric = torch.zeros(batch, length, examples[0].lyric.shape[-1])
    chorus = torch.zeros(batch, length, dtype=torch.bool)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    target = torch.stack([example.target for example in examples])
    for index, example in enumerate(examples):
        size = len(example.melody)
        melody[index, :size] = example.melody
        lyric[index, :size] = example.lyric
        chorus[index, :size] = example.chorus
        mask[index, :size] = True
    return melody, lyric, chorus, mask, target


def group_batches(examples, size, rng):
    order = list(examples)
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), size):
        yield order[start : start + size]


def predict(
    model: nn.Module, examples: list[Example], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for group in group_batches(examples, 64, None):
            melody, lyric, chorus, mask, target = collate(group)
            output = model(
                melody.to(device),
                lyric.to(device),
                chorus.to(device),
                mask.to(device),
            )
            predictions.append(output.cpu().numpy())
            targets.append(target.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def validation_loss(model, examples, criterion, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for group in group_batches(examples, 64, None):
            melody, lyric, chorus, mask, target = collate(group)
            output = model(
                melody.to(device),
                lyric.to(device),
                chorus.to(device),
                mask.to(device),
            )
            total += criterion(output, target.to(device)).item() * len(group)
    return total / max(len(examples), 1)


def run(config: ProjectConfig, paths: ProjectPaths) -> dict[str, float]:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    examples = load_examples(config, paths)
    if len(examples) < 3:
        raise RuntimeError(f"need at least 3 labelled tracks, found {len(examples)}")
    training = config.training
    train_set, val_set, test_set = split_examples(examples, config)
    logger.info(
        "train %d / val %d / test %d",
        len(train_set),
        len(val_set),
        len(test_set),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_emotion_regressor(
        config, examples[0].melody.shape[-1], examples[0].lyric.shape[-1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)
    criterion = nn.MSELoss()
    rng = random.Random(config.seed)

    checkpoint = paths.checkpoints / f"{config.data.dataset}.pt"
    writer = SummaryWriter(
        paths.artifacts / "logs" / "tensorboard" / config.data.dataset
    )
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, training.epochs + 1):
        model.train()
        total = 0.0
        for group in group_batches(train_set, training.batch_size, rng):
            optimizer.zero_grad()
            melody, lyric, chorus, mask, target = collate(group)
            output = model(
                melody.to(device),
                lyric.to(device),
                chorus.to(device),
                mask.to(device),
            )
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(group)
        train_loss = total / len(train_set)
        validation = validation_loss(model, val_set, criterion, device)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", validation, epoch)
        logger.info("epoch %d train %.4f val %.4f", epoch, train_loss, validation)
        if validation < best_loss - 1e-6:
            best_loss = validation
            stale = 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= training.patience:
                logger.info("early stop at epoch %d", epoch)
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    if test_set:
        predictions, targets = predict(model, test_set, device)
        test_metrics = evaluate(predictions, targets)
        logger.info("test metrics: %s", test_metrics)
        for name, value in test_metrics.items():
            writer.add_scalar(f"test/{name}", value, 0)
    predictions, targets = predict(model, val_set, device)
    metrics = evaluate(predictions, targets)
    for name, value in metrics.items():
        writer.add_scalar(f"val/{name}", value, 0)
    writer.close()
    return metrics
