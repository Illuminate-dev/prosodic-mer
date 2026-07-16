import logging
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from spleeter.separator import Separator

from pmr.config import ProjectConfig
from pmr.paths import ProjectPaths

logger = logging.getLogger(__name__)

VOCAL_STEM = "vocals"
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aif", ".aiff", ".wma")


class SeparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeparationResult:
    source: Path
    target: Path
    duration_s: float


class VocalSeparator:
    def __init__(self, stems):
        logger.info("loading spleeter: %s", stems)
        self._separator = Separator(f"spleeter:{stems}")

    def separate(self, source: Path) -> tuple[np.ndarray, int]:
        with tempfile.TemporaryDirectory(prefix="spleeter-") as workdir:
            destination = Path(workdir)
            self._separator.separate_to_file(
                str(source),
                str(destination),
                filename_format="{filename}/{instrument}.{codec}",
                codec="wav",
                duration=None,
                synchronous=True,
            )
            stem_path = destination / source.stem / f"{VOCAL_STEM}.wav"
            if not stem_path.exists():
                produced = sorted(path.name for path in destination.rglob("*"))
                raise SeparationError(
                    f"spleeter produced {produced}, expected {stem_path.name}"
                )
            samples, sample_rate = sf.read(str(stem_path), always_2d=True)
        return samples, int(sample_rate)


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if samples.ndim != 2:
        raise ValueError(f"expected 1-2 dimensions, got shape {samples.shape}")
    return samples.mean(axis=1)


def write_vocals(samples: np.ndarray, sample_rate: int, target: Path) -> float:
    waveform = to_mono(samples)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), waveform, sample_rate, subtype="PCM_16")
    return len(waveform) / sample_rate


def discover_tracks(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def separate_track(
    separator: VocalSeparator, source: Path, target: Path
) -> SeparationResult:
    samples, sample_rate = separator.separate(source)
    duration = write_vocals(samples, sample_rate, target)
    return SeparationResult(source=source, target=target, duration_s=duration)


def separate_dataset(
    config: ProjectConfig,
    paths: ProjectPaths,
    track_ids: Iterable[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[SeparationResult]:
    dataset = config.data.dataset
    raw_dir = paths.raw_dataset(dataset)
    target_dir = paths.vocals(dataset)

    tracks = discover_tracks(raw_dir)
    if track_ids is not None:
        wanted = set(track_ids)
        tracks = [track for track in tracks if track.stem in wanted]
    if limit is not None:
        tracks = tracks[:limit]

    if not tracks:
        logger.warning("no tracks found under %s", raw_dir)
        return []

    paths.ensure_outputs()
    separator = VocalSeparator(config.data.separation.spleeter_stems)
    logger.info("separating %d track(s)", len(tracks))

    results: list[SeparationResult] = []
    for index, source in enumerate(tracks, start=1):
        target = target_dir / f"{source.stem}.wav"
        if target.exists() and not overwrite:
            logger.debug("skipping existing %s", target.name)
            continue
        try:
            result = separate_track(separator, source, target)
        except Exception as error:
            logger.error(
                "[%d/%d] %s failed: %s", index, len(tracks), source.name, error
            )
            continue
        logger.info(
            "[%d/%d] %s -> %s (%.1fs)",
            index,
            len(tracks),
            source.name,
            target.name,
            result.duration_s,
        )
        results.append(result)

    logger.info("wrote %d to %s", len(results), target_dir)
    return results
