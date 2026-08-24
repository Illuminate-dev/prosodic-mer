import json
import logging
import random
from dataclasses import dataclass, replace

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
    track_target: torch.Tensor
    prosody: torch.Tensor | None = None
    sentence_targets: torch.Tensor | None = None
    unit_mask: torch.Tensor | None = None
    segment: torch.Tensor | None = None


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


def load_words(paths: ProjectPaths, dataset: str, track_id: str):
    source = paths.transcriptions(dataset) / f"{track_id}.json"
    segments = json.loads(source.read_text())["segments"]
    words = [word for segment in segments for word in segment["words"]]
    pair_of_word = np.repeat(
        np.arange(len(segments)), [len(segment["words"]) for segment in segments]
    )
    starts = np.array([word["start"] for word in words], dtype="float32")
    ends = np.array([word["end"] for word in words], dtype="float32")
    return pair_of_word, starts, ends


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
    parts = [config.data.dataset]
    if config.model.processing_level == "word":
        parts.append("word")
    if config.model.prosody_level:
        parts.append(config.model.prosody_level)
    if config.model.supervision_level == "sentence":
        parts.append("sent")
    return "_".join(parts)


def load_examples(config: ProjectConfig, paths: ProjectPaths) -> list[Example]:
    dataset = config.data.dataset
    level = config.model.processing_level
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
        track_target = np.nanmean(values, axis=0)
        if np.isnan(track_target).any():
            logger.warning("skipping %s (no annotated units)", track_id)
            skipped += 1
            continue

        pair_of_word, word_starts, word_ends = load_words(
            paths, dataset, track_id
        )
        if level == "word":
            melody_features = melody_features[pair_of_word]
            lyric_features = lyric_features[pair_of_word]
            starts, ends = word_starts, word_ends

        prosody_features = None
        if config.model.prosody_level:
            prosody = load_feature(
                paths, dataset, "word", config.model.prosody_level, track_id
            )
            if prosody is None:
                logger.warning("skipping %s (missing prosody)", track_id)
                skipped += 1
                continue
            prosody_features = fill_missing(prosody[0])
            if len(pair_of_word) != len(prosody_features):
                logger.warning("skipping %s (prosody misaligned)", track_id)
                skipped += 1
                continue
            if level == "pair":
                prosody_features = pool_prosody(
                    prosody_features, pair_of_word, len(melody_features)
                )

        sentence_targets = None
        unit_mask = None
        segment = None
        if config.model.supervision_level == "sentence":
            sentence_targets = np.nan_to_num(values, nan=0.0)
            unit_mask = ~np.isnan(values).any(axis=1)
            segment = pair_of_word if level == "word" else np.arange(len(values))

        examples.append(
            Example(
                track_id=track_id,
                melody=torch.tensor(melody_features, dtype=torch.float32),
                lyric=torch.tensor(lyric_features, dtype=torch.float32),
                chorus=load_chorus(paths, dataset, track_id, starts, ends),
                track_target=torch.tensor(track_target, dtype=torch.float32),
                prosody=(
                    torch.tensor(prosody_features, dtype=torch.float32)
                    if prosody_features is not None
                    else None
                ),
                sentence_targets=(
                    torch.tensor(sentence_targets, dtype=torch.float32)
                    if sentence_targets is not None
                    else None
                ),
                unit_mask=(
                    torch.tensor(unit_mask, dtype=torch.bool)
                    if unit_mask is not None
                    else None
                ),
                segment=(
                    torch.tensor(segment, dtype=torch.long)
                    if segment is not None
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


def fold_split(
    examples: list[Example], config: ProjectConfig, fold: int, folds: int
) -> tuple[list[Example], list[Example], list[Example]]:
    by_id = {example.track_id: example for example in examples}
    ids = sorted(by_id)
    random.Random(config.seed).shuffle(ids)
    chunks = [list(chunk) for chunk in np.array_split(ids, folds)]
    test = chunks[fold]
    rest = [
        identifier
        for index, chunk in enumerate(chunks)
        if index != fold
        for identifier in chunk
    ]
    val_count = int(len(rest) * config.training.val_fraction)
    return (
        [by_id[i] for i in rest[val_count:]],
        [by_id[i] for i in rest[:val_count]],
        [by_id[i] for i in test],
    )


def collate(examples: list[Example]):
    length = max(len(example.melody) for example in examples)
    batch = len(examples)
    melody = torch.zeros(batch, length, examples[0].melody.shape[-1])
    lyric = torch.zeros(batch, length, examples[0].lyric.shape[-1])
    chorus = torch.zeros(batch, length, dtype=torch.bool)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    track_target = torch.stack([example.track_target for example in examples])

    prosody = None
    if examples[0].prosody is not None:
        prosody = torch.zeros(batch, length, examples[0].prosody.shape[-1])

    segment = sentence_targets = unit_mask = None
    if examples[0].sentence_targets is not None:
        segments = max(len(example.sentence_targets) for example in examples)
        segment = torch.zeros(batch, length, dtype=torch.long)
        sentence_targets = torch.zeros(
            batch, segments, examples[0].sentence_targets.shape[-1]
        )
        unit_mask = torch.zeros(batch, segments, dtype=torch.bool)

    for index, example in enumerate(examples):
        size = len(example.melody)
        melody[index, :size] = example.melody
        lyric[index, :size] = example.lyric
        chorus[index, :size] = example.chorus
        mask[index, :size] = True
        if prosody is not None:
            prosody[index, :size] = example.prosody
        if segment is not None:
            segment[index, :size] = example.segment
            count = len(example.sentence_targets)
            sentence_targets[index, :count] = example.sentence_targets
            unit_mask[index, :count] = example.unit_mask

    return (
        melody,
        lyric,
        chorus,
        mask,
        track_target,
        prosody,
        segment,
        sentence_targets,
        unit_mask,
    )


def standardize_prosody(*splits: list[Example]):
    rows = torch.cat([example.prosody for example in splits[0]], dim=0)
    mean = rows.mean(dim=0)
    std = rows.std(dim=0).clamp(min=1e-6)
    return tuple(
        [
            replace(example, prosody=(example.prosody - mean) / std)
            for example in split
        ]
        for split in splits
    )


def group_batches(examples, size, rng):
    order = list(examples)
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), size):
        yield order[start : start + size]


def masked_mse(output, target, unit_mask):
    squared = ((output - target) ** 2).mean(dim=-1)
    weights = unit_mask.to(squared.dtype)
    return (squared * weights).sum() / weights.sum().clamp(min=1.0)


def segment_mean(values, segment, mask, segments):
    weights = mask.unsqueeze(-1).to(values.dtype)
    pooled = values.new_zeros(values.shape[0], segments, values.shape[-1])
    counts = values.new_zeros(values.shape[0], segments)
    pooled.scatter_add_(
        1,
        segment.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
        values * weights,
    )
    counts.scatter_add_(1, segment, mask.to(values.dtype))
    return pooled / counts.clamp(min=1.0).unsqueeze(-1)


def sentence_loss(output, sentence_targets, unit_mask, segment, mask):
    pooled = segment_mean(output, segment, mask, sentence_targets.shape[1])
    return masked_mse(pooled, sentence_targets, unit_mask)


def batch_loss(model, group, criterion, device, supervision):
    (
        melody,
        lyric,
        chorus,
        mask,
        track_target,
        prosody,
        segment,
        sentence_targets,
        unit_mask,
    ) = collate(group)
    melody, lyric, chorus, mask = (
        tensor.to(device) for tensor in (melody, lyric, chorus, mask)
    )
    pooled = supervision == "track"
    if prosody is not None:
        output = model(
            melody, lyric, chorus, mask, prosody.to(device), pooled=pooled
        )
    else:
        output = model(melody, lyric, chorus, mask, pooled=pooled)
    if supervision == "track":
        return criterion(output, track_target.to(device))
    return sentence_loss(
        output,
        sentence_targets.to(device),
        unit_mask.to(device),
        segment.to(device),
        mask,
    )


def predict(
    model: nn.Module,
    examples: list[Example],
    device: torch.device,
    supervision: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, track_targets = [], []
    with torch.no_grad():
        for group in group_batches(examples, 64, None):
            (
                melody,
                lyric,
                chorus,
                mask,
                track_target,
                prosody,
                segment,
                _,
                unit_mask,
            ) = collate(group)
            melody, lyric, chorus, mask = (
                tensor.to(device) for tensor in (melody, lyric, chorus, mask)
            )
            pooled = supervision == "track"
            if prosody is not None:
                output = model(
                    melody,
                    lyric,
                    chorus,
                    mask,
                    prosody.to(device),
                    pooled=pooled,
                )
            else:
                output = model(melody, lyric, chorus, mask, pooled=pooled)
            if supervision == "sentence":
                output = segment_mean(
                    output, segment.to(device), mask, unit_mask.shape[1]
                )
                weights = unit_mask.unsqueeze(-1).to(output.dtype).to(device)
                output = (output * weights).sum(1) / weights.sum(1).clamp(min=1.0)
            predictions.append(output.cpu().numpy())
            track_targets.append(track_target.numpy())
    return np.concatenate(predictions), np.concatenate(track_targets)


def validation_loss(model, examples, criterion, device, supervision) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for group in group_batches(examples, 64, None):
            loss = batch_loss(model, group, criterion, device, supervision)
            total += loss.item() * len(group)
    return total / max(len(examples), 1)


def fit(
    config: ProjectConfig,
    paths: ProjectPaths,
    train_set: list[Example],
    val_set: list[Example],
    test_set: list[Example],
    name: str,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    training = config.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prosody_dim = (
        train_set[0].prosody.shape[-1] if config.model.prosody_level else None
    )
    model = build_emotion_regressor(
        config,
        train_set[0].melody.shape[-1],
        train_set[0].lyric.shape[-1],
        prosody_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)
    criterion = nn.MSELoss()
    rng = random.Random(seed)

    checkpoint = paths.checkpoints / f"{name}.pt"
    writer = SummaryWriter(paths.artifacts / "logs" / "tensorboard" / name)
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, training.epochs + 1):
        model.train()
        total = 0.0
        for group in group_batches(train_set, training.batch_size, rng):
            optimizer.zero_grad()
            loss = batch_loss(
                model, group, criterion, device, config.model.supervision_level
            )
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(group)
        train_loss = total / len(train_set)
        validation = validation_loss(
            model, val_set, criterion, device, config.model.supervision_level
        )
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
    test_metrics: dict[str, float] = {}
    if test_set:
        predictions, track_targets = predict(
            model, test_set, device, config.model.supervision_level
        )
        test_metrics = evaluate(predictions, track_targets)
        logger.info("test metrics: %s", test_metrics)
        for key, value in test_metrics.items():
            writer.add_scalar(f"test/{key}", value, 0)
    predictions, track_targets = predict(
        model, val_set, device, config.model.supervision_level
    )
    metrics = evaluate(predictions, track_targets)
    for key, value in metrics.items():
        writer.add_scalar(f"val/{key}", value, 0)
    writer.close()
    return metrics, test_metrics


def run(config: ProjectConfig, paths: ProjectPaths) -> dict[str, float]:
    examples = load_examples(config, paths)
    if len(examples) < 3:
        raise RuntimeError(f"need at least 3 labelled tracks, found {len(examples)}")
    train_set, val_set, test_set = split_examples(examples, config)
    if config.model.prosody_level:
        train_set, val_set, test_set = standardize_prosody(
            train_set, val_set, test_set
        )
    logger.info(
        "train %d / val %d / test %d",
        len(train_set),
        len(val_set),
        len(test_set),
    )
    metrics, _ = fit(
        config, paths, train_set, val_set, test_set, run_name(config), config.seed
    )
    return metrics
