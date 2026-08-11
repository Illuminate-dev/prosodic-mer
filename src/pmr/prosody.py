import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from pmr.config import ProjectConfig
from pmr.features import Unit, load_units, write_features
from pmr.paths import ProjectPaths

logger = logging.getLogger(__name__)

PITCH_FLOOR = 75.0
PITCH_CEILING = 600.0
TIME_STEP = 0.01
FORMANT_COUNT = 5
MAXIMUM_FORMANT = 5500.0
SHORTEST_PERIOD = 0.0001
LONGEST_PERIOD = 0.02
MAX_PERIOD_FACTOR = 1.3
MAX_AMPLITUDE_FACTOR = 1.6

PROSODY_FEATURES = (
    "f0_mean",
    "f0_std",
    "f0_min",
    "f0_max",
    "f0_range",
    "f0_slope",
    "intensity_mean",
    "intensity_std",
    "intensity_min",
    "intensity_max",
    "f1_mean",
    "f2_mean",
    "f3_mean",
    "f1_std",
    "f2_std",
    "jitter_local",
    "jitter_rap",
    "jitter_ppq5",
    "shimmer_local",
    "shimmer_apq3",
    "shimmer_apq5",
    "hnr_mean",
    "duration",
    "voiced_duration",
    "voiced_fraction",
)
PROSODY_DIM = len(PROSODY_FEATURES)


class ProsodyExtractor:
    def extract(self, source: Path, units: list[Unit]) -> np.ndarray:
        sound = parselmouth.Sound(str(source))
        rows = []
        for unit in units:
            try:
                part = sound.extract_part(from_time=unit.start, to_time=unit.end)
                rows.append(self._unit(part))
            except Exception as error:
                logger.debug("prosody failed for '%s': %s", unit.text, error)
                rows.append(np.full(PROSODY_DIM, np.nan, dtype="float32"))
        return np.stack(rows).astype("float32")

    def _unit(self, part: parselmouth.Sound) -> np.ndarray:
        duration = float(part.xmax - part.xmin)
        pitch = part.to_pitch(
            time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING
        )
        frequency = pitch.selected_array["frequency"]
        voiced = frequency > 0
        if voiced.any():
            f0 = frequency[voiced]
            semitones = 12.0 * np.log2(f0)
            slope = (
                float(np.polyfit(pitch.xs()[voiced], semitones, 1)[0])
                if f0.size > 1
                else 0.0
            )
            pitch_stats = [
                f0.mean(),
                f0.std(),
                f0.min(),
                f0.max(),
                f0.max() - f0.min(),
                slope,
            ]
        else:
            pitch_stats = [np.nan] * 6
        voiced_fraction = float(voiced.mean()) if voiced.size else 0.0

        intensity = part.to_intensity(time_step=TIME_STEP).values.flatten()

        formants = part.to_formant_burg(
            max_number_of_formants=FORMANT_COUNT, maximum_formant=MAXIMUM_FORMANT
        )
        formant_stats = [
            call(formants, "Get mean", number, part.xmin, part.xmax, "hertz")
            for number in (1, 2, 3)
        ] + [
            call(
                formants,
                "Get standard deviation",
                number,
                part.xmin,
                part.xmax,
                "hertz",
            )
            for number in (1, 2)
        ]

        point_process = call(
            part, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING
        )
        jitter = [
            call(
                point_process,
                f"Get jitter ({name})",
                0,
                0,
                SHORTEST_PERIOD,
                LONGEST_PERIOD,
                MAX_PERIOD_FACTOR,
            )
            for name in ("local", "rap", "ppq5")
        ]
        shimmer = [
            call(
                [part, point_process],
                f"Get shimmer ({name})",
                0,
                0,
                SHORTEST_PERIOD,
                LONGEST_PERIOD,
                MAX_PERIOD_FACTOR,
                MAX_AMPLITUDE_FACTOR,
            )
            for name in ("local", "apq3", "apq5")
        ]
        harmonicity = call(
            part, "To Harmonicity (cc)", TIME_STEP, PITCH_FLOOR, 0.1, 1.0
        )
        hnr = call(harmonicity, "Get mean", 0, 0)

        return np.array(
            [
                *pitch_stats,
                *_stats(intensity),
                *formant_stats,
                *jitter,
                *shimmer,
                hnr,
                duration,
                voiced_fraction * duration,
                voiced_fraction,
            ],
            dtype="float32",
        )


def _stats(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [np.nan] * 4
    return [values.mean(), values.std(), values.min(), values.max()]


def prosody_dataset(
    config: ProjectConfig,
    paths: ProjectPaths,
    track_ids: Iterable[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    dataset = config.data.dataset
    vocal_dir = paths.vocals(dataset)
    target_dir = paths.features(dataset, "word", "prosody")

    tracks = sorted(paths.transcriptions(dataset).glob("*.json"))
    if track_ids is not None:
        wanted = set(track_ids)
        tracks = [path for path in tracks if path.stem in wanted]
    if limit is not None:
        tracks = tracks[:limit]

    if not tracks:
        logger.warning("no transcriptions found for %s", dataset)
        return []

    extractor = ProsodyExtractor()
    done: list[str] = []
    skipped = 0
    for index, track in enumerate(tracks, start=1):
        track_id = track.stem
        target = target_dir / f"{track_id}.npz"
        if target.exists() and not overwrite:
            logger.debug("skipping existing %s", track_id)
            continue
        source = vocal_dir / f"{track_id}.wav"
        units = load_units(config, paths, track_id, unit="word")
        if not source.exists() or not units:
            logger.warning("[%d/%d] %s has no words", index, len(tracks), track_id)
            skipped += 1
            continue
        try:
            features = extractor.extract(source, units)
        except Exception as error:
            logger.error("[%d/%d] %s failed: %s", index, len(tracks), track_id, error)
            skipped += 1
            continue
        write_features(target, features, units)
        logger.info("[%d/%d] %s -> %d words", index, len(tracks), track_id, len(units))
        done.append(track_id)

    logger.info("wrote %d track(s) to %s (%d skipped)", len(done), target_dir, skipped)
    return done
