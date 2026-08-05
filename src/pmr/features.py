import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchvggish import vggish, waveform_to_examples
from transformers import AlbertModel, AlbertTokenizerFast

from pmr.config import ProjectConfig
from pmr.paths import ProjectPaths

logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aif", ".aiff")
ALBERT_NAME = "albert-base-v2"
ALBERT_MAX_TOKENS = 128
ALBERT_BATCH = 64
VGGISH_HOP_S = 0.96


@dataclass(frozen=True)
class Unit:
    text: str
    start: float
    end: float


class VggishEmbedder:
    def __init__(self, device: str):
        self._device = torch.device(device)
        self._model = vggish(postprocess=False).to(self._device).eval()

    def embed(self, source: Path, units: list[Unit]) -> np.ndarray:
        data, rate = sf.read(str(source), dtype="float32")
        examples = waveform_to_examples(data, rate).to(self._device)
        with torch.no_grad():
            embeddings = self._model(examples).cpu().numpy()
        starts = np.arange(len(embeddings)) * VGGISH_HOP_S
        ends = starts + VGGISH_HOP_S
        return np.stack(
            [_pool(embeddings, starts, ends, unit) for unit in units]
        ).astype("float32")


class AlbertEmbedder:
    def __init__(self, device: str):
        self._device = torch.device(device)
        self._tokenizer = AlbertTokenizerFast.from_pretrained(ALBERT_NAME)
        self._model = AlbertModel.from_pretrained(ALBERT_NAME).to(self._device).eval()

    def embed(self, units: list[Unit]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for offset in range(0, len(units), ALBERT_BATCH):
            texts = [unit.text for unit in units[offset : offset + ALBERT_BATCH]]
            batches.append(self._embed_batch(texts))
        return np.concatenate(batches).astype("float32")

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=ALBERT_MAX_TOKENS,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask").to(self._device)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = self._model(**encoded).last_hidden_state
        mask = (encoded["attention_mask"] * (1 - special)).unsqueeze(-1).float()
        return ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)).cpu().numpy()


def _pool(
    embeddings: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    unit: Unit,
) -> np.ndarray:
    selected = (starts < unit.end) & (ends > unit.start)
    if not selected.any():
        centres = (starts + ends) / 2
        selected = np.zeros(len(embeddings), dtype=bool)
        selected[np.argmin(np.abs(centres - (unit.start + unit.end) / 2))] = True
    return embeddings[selected].mean(axis=0)


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_audio(raw_dir: Path, track_id: str) -> Path | None:
    for suffix in AUDIO_SUFFIXES:
        candidate = raw_dir / f"{track_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_units(
    config: ProjectConfig, paths: ProjectPaths, track_id: str
) -> list[Unit]:
    dataset = config.data.dataset
    if config.data.features.unit == "sentence":
        source = paths.alignments(dataset) / f"{track_id}.json"
        pairs = json.loads(source.read_text())["pairs"]
        return [
            Unit(
                text=pair["lyric"]["text"],
                start=pair["melody"]["start"],
                end=pair["melody"]["end"],
            )
            for pair in pairs
        ]
    source = paths.transcriptions(dataset) / f"{track_id}.json"
    segments = json.loads(source.read_text())["segments"]
    return [
        Unit(text=word["text"], start=word["start"], end=word["end"])
        for segment in segments
        for word in segment["words"]
    ]


def write_features(target: Path, features: np.ndarray, units: list[Unit]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        features=features,
        start=np.array([unit.start for unit in units], dtype="float32"),
        end=np.array([unit.end for unit in units], dtype="float32"),
        text=np.array([unit.text for unit in units]),
    )


def feature_dataset(
    config: ProjectConfig,
    paths: ProjectPaths,
    track_ids: Iterable[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    dataset = config.data.dataset
    raw_dir = paths.raw_dataset(dataset)
    alignment_dir = paths.alignments(dataset)
    vggish_dir = paths.features(dataset, config.data.features.unit, "vggish")
    albert_dir = paths.features(dataset, config.data.features.unit, "albert")

    tracks = sorted(alignment_dir.glob("*.json")) if alignment_dir.exists() else []
    if track_ids is not None:
        wanted = set(track_ids)
        tracks = [path for path in tracks if path.stem in wanted]
    if limit is not None:
        tracks = tracks[:limit]

    if not tracks:
        logger.warning("no alignments found under %s", alignment_dir)
        return []

    device = resolve_device()
    logger.info("extracting features on %s", device)
    vggish_embedder = VggishEmbedder(device)
    albert_embedder = AlbertEmbedder(device)

    track_ids_done: list[str] = []
    skipped = 0
    for index, alignment in enumerate(tracks, start=1):
        track_id = alignment.stem
        vggish_target = vggish_dir / f"{track_id}.npz"
        albert_target = albert_dir / f"{track_id}.npz"
        if (
            vggish_target.exists()
            and albert_target.exists()
            and not overwrite
        ):
            logger.debug("skipping existing %s", track_id)
            continue
        audio = find_audio(raw_dir, track_id)
        if audio is None:
            logger.error("[%d/%d] %s has no raw audio", index, len(tracks), track_id)
            skipped += 1
            continue
        units = load_units(config, paths, track_id)
        if not units:
            logger.warning("[%d/%d] %s has no units", index, len(tracks), track_id)
            skipped += 1
            continue
        try:
            vggish_features = vggish_embedder.embed(audio, units)
            albert_features = albert_embedder.embed(units)
        except Exception as error:
            logger.error(
                "[%d/%d] %s failed: %s", index, len(tracks), track_id, error
            )
            skipped += 1
            continue
        write_features(vggish_target, vggish_features, units)
        write_features(albert_target, albert_features, units)
        logger.info(
            "[%d/%d] %s -> %d units", index, len(tracks), track_id, len(units)
        )
        track_ids_done.append(track_id)

    logger.info(
        "wrote %d track(s) to %s (%d skipped)",
        len(track_ids_done),
        vggish_dir.parent,
        skipped,
    )
    return track_ids_done
