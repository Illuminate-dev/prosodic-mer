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
    prosody: torch.Tensor | None = None


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


def load_pair_of_word(
    paths: ProjectPaths, dataset: str, track_id: str
) -> np.ndarray:
    source = paths.transcriptions(dataset) / f"{track_id}.json"
    segments = json.loads(source.read_text())["segments"]
    return np.repeat(
        np.arange(len(segments)), [len(segment["words"]) for segment in segments]
    )


def fill_missing(features: np.ndarray) -> np.ndarray:
    column_mean = np.nan_to_num(np.nanmean(features, axis=0), nan=0.0)
    missing = np.isnan(features)
    if not missing.any():
        return features
    filled = features.copy()
    filled[missing] = np.take(column_mean, np.where(missing)[1])
    return filled


def pool_prosody(
    features: np.ndarray, pair_of_word: np.ndarray, pairs: int
) -> np.ndarray:
    pooled = np.zeros((pairs, features.shape[1]), dtype="float32")
    counts = np.zeros(pairs, dtype="float32")
    np.add.at(pooled, pair_of_word, features)
    np.add.at(counts, pair_of_word, 1)
    return pooled / np.maximum(counts, 1)[:, None]


def run_name(config: ProjectConfig) -> str:
    dataset = config.data.dataset
    if not config.model.prosody:
        return dataset
    return f"{dataset}_prosody-{config.model.level}"


def load_examples(config: ProjectConfig, paths: ProjectPaths) -> list[Example]:
    dataset = config.data.dataset
    level = config.model.level
    label_dir = paths.labels(dataset)
    label_paths = sorted(label_dir.glob("*.npz")) if label_dir.exists() else []

    examples: list[Example] = []
    skipped = 0
    for label_path in label_paths:
        track_id = label_path.stem
        melody = load_feature(paths, dataset, "sentence", "vggish", track_id)
        lyric = load_feature(paths, dataset, "sentence", "albert", track_id)
        if melody is None or lyric is None:
            logger.warning("skipping %s (missing features)", track_id)
            skipped += 1
            continue
        melody_features, starts, ends = melody
        lyric_features = lyric[0]
        values = np.load(label_path)["values"]
        if len(values) != len(melody_features) or len(lyric_features) != len(
            melody_features
        ):
            logger.warning("skipping %s (length mismatch)", track_id)
            skipped += 1
            continue
        target = np.nanmean(values, axis=0)
        if np.isnan(target).any():
            logger.warning("skipping %s (no annotated units)", track_id)
            skipped += 1
            continue

        prosody_features = None
        if config.model.prosody:
            prosody = load_feature(paths, dataset, "word", "prosody", track_id)
            if prosody is None:
                logger.warning("skipping %s (missing prosody)", track_id)
                skipped += 1
                continue
            pair_of_word = load_pair_of_word(paths, dataset, track_id)
            prosody_features = fill_missing(prosody[0])
            if len(pair_of_word) != len(prosody_features):
                logger.warning("skipping %s (prosody misaligned)", track_id)
                skipped += 1
                continue
            if level == "word":
                melody_features = melody_features[pair_of_word]
                lyric_features = lyric_features[pair_of_word]
                starts, ends = prosody[1], prosody[2]
            else:
                prosody_features = pool_prosody(
                    prosody_features, pair_of_word, len(melody_features)
                )

        examples.append(
            Example(
                track_id=track_id,
                melody=torch.tensor(melody_features, dtype=torch.float32),
                lyric=torch.tensor(lyric_features, dtype=torch.float32),
                chorus=load_chorus(paths, dataset, track_id, starts, ends),
                target=torch.tensor(target, dtype=torch.float32),
                prosody=(
                    torch.tensor(prosody_features, dtype=torch.float32)
                    if prosody_features is not None
                    else None
                ),
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
    prosody = None
    if examples[0].prosody is not None:
        prosody = torch.zeros(batch, length, examples[0].prosody.shape[-1])
    for index, example in enumerate(examples):
        size = len(example.melody)
        melody[index, :size] = example.melody
        lyric[index, :size] = example.lyric
        chorus[index, :size] = example.chorus
        mask[index, :size] = True
        if prosody is not None:
            prosody[index, :size] = example.prosody
    return melody, lyric, chorus, mask, target, prosody


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
            melody, lyric, chorus, mask, target, prosody = collate(group)
            output = model(
                melody.to(device),
                lyric.to(device),
                chorus.to(device),
                mask.to(device),
                prosody.to(device) if prosody is not None else None,
            )
            predictions.append(output.cpu().numpy())
            targets.append(target.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def validation_loss(model, examples, criterion, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for group in group_batches(examples, 64, None):
            melody, lyric, chorus, mask, target, prosody = collate(group)
            output = model(
                melody.to(device),
                lyric.to(device),
                chorus.to(device),
                mask.to(device),
                prosody.to(device) if prosody is not None else None,
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
    prosody_dim = examples[0].prosody.shape[-1] if config.model.prosody else None
    model = build_emotion_regressor(
        config,
        examples[0].melody.shape[-1],
        examples[0].lyric.shape[-1],
        prosody_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)
    criterion = nn.MSELoss()
    rng = random.Random(config.seed)

    name = run_name(config)
    checkpoint = paths.checkpoints / f"{name}.pt"
    writer = SummaryWriter(paths.artifacts / "logs" / "tensorboard" / name)
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, training.epochs + 1):
        model.train()
        total = 0.0
        for group in group_batches(train_set, training.batch_size, rng):
            optimizer.zero_grad()
            melody, lyric, chorus, mask, target, prosody = collate(group)
            output = model(
                melody.to(device),
                lyric.to(device),
                chorus.to(device),
                mask.to(device),
                prosody.to(device) if prosody is not None else None,
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
