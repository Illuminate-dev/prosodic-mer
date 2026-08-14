import csv
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from pmr.config import ProjectConfig
from pmr.paths import ProjectPaths

logger = logging.getLogger(__name__)

DYNAMIC = Path(
    "annotations/annotations averaged per song/dynamic (per second annotations)"
)
SAMPLE = "sample_"


def load_dynamic(source: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    dynamic: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with source.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        times = np.array(
            [int(name[len(SAMPLE) : -2]) / 1000.0 for name in header[1:]]
        )
        for row in reader:
            if not row:
                continue
            values = np.array([float(value) for value in row[1:]])
            dynamic[int(row[0])] = (times[: len(values)], values)
    return dynamic


def span_mean(
    times: np.ndarray,
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    result = np.full(len(starts), np.nan)
    for index, (start, end) in enumerate(zip(starts, ends)):
        low = max(start, times[0])
        high = min(end, times[-1])
        if low > high:
            continue
        inside = values[(times >= low) & (times <= high)]
        result[index] = (
            inside.mean() if inside.size else np.interp((low + high) / 2, times, values)
        )
    return result


def label_dataset(
    config: ProjectConfig,
    paths: ProjectPaths,
    track_ids: Iterable[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    dataset = config.data.dataset
    dynamic_dir = paths.raw_dataset(dataset) / DYNAMIC
    valence = load_dynamic(dynamic_dir / "valence.csv")
    arousal = load_dynamic(dynamic_dir / "arousal.csv")

    feature_dir = paths.features(dataset, "sentence", "vggish")
    target_dir = paths.labels(dataset)
    tracks = sorted(feature_dir.glob("*.npz")) if feature_dir.exists() else []
    if track_ids is not None:
        wanted = set(track_ids)
        tracks = [path for path in tracks if path.stem in wanted]
    if limit is not None:
        tracks = tracks[:limit]

    if not tracks:
        logger.warning("no features found under %s", feature_dir)
        return []

    logger.info("labelling %d tracks", len(tracks))

    results: list[str] = []
    skipped = 0
    for index, feature in enumerate(tracks, start=1):
        track_id = feature.stem
        target = target_dir / f"{track_id}.npz"
        if target.exists() and not overwrite:
            logger.debug("skipping existing %s", track_id)
            continue
        try:
            song = int(track_id)
        except ValueError:
            logger.warning("[%d/%d] %s is not a DEAM id", index, len(tracks), track_id)
            skipped += 1
            continue
        if song not in valence or song not in arousal:
            logger.warning("[%d/%d] %s has no annotations", index, len(tracks), track_id)
            skipped += 1
            continue
        data = np.load(feature)
        starts, ends = data["start"], data["end"]
        values = np.stack(
            [
                span_mean(*valence[song], starts, ends),
                span_mean(*arousal[song], starts, ends),
            ],
            axis=1,
        ).astype("float32")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, values=values)
        logger.info(
            "[%d/%d] %s -> %d units", index, len(tracks), track_id, len(values)
        )
        results.append(track_id)

    logger.info("wrote %d to %s (%d skipped)", len(results), target_dir, skipped)
    return results
