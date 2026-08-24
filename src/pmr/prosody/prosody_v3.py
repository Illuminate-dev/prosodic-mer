import logging
from collections.abc import Iterable
from pathlib import Path

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from pmr.config import ProjectConfig
from pmr.features import Unit, find_audio, load_units, write_features
from pmr.paths import ProjectPaths
from pmr.prosody.common import (
    HOP_LENGTH,
    PITCH_CEILING,
    PITCH_FLOOR,
    SAMPLE_RATE,
    TIME_STEP,
    beat_residual,
    chord_and_tension,
    relative_st,
    track_key,
)

logger = logging.getLogger(__name__)

PHRASE_GAP_S = 0.4
MIN_DUR_S = 0.05

CLIP = {
    "f0_rel_st": (-12.0, 12.0),
    "f0_iqr_st": (0.0, 12.0),
    "f0_range_st": (0.0, 12.0),
    "f0_slope": (-50.0, 50.0),
    "d_f0_rel": (-12.0, 12.0),
    "speech_rate": (0.0, 30.0),
    "dur": (0.0, 3.0),
    "gap_prev": (0.0, 2.0),
    "onset_rate": (0.0, 10.0),
    "loud_rel": (-30.0, 30.0),
    "loud_range": (0.0, 60.0),
    "loud_change_mag": (0.0, 10.0),
    "d_loud": (-30.0, 30.0),
    "hnr_rel": (-15.0, 15.0),
    "jitter_local": (0.0, 0.05),
    "phrase_f0": (-12.0, 12.0),
    "phrase_loud": (-30.0, 30.0),
}

PROSODY_V3_FEATURES = (
    "f0_rel_st",
    "f0_iqr_st",
    "f0_range_st",
    "f0_slope",
    "voice_dir",
    "voice_mobility",
    "d_f0_rel",
    "speech_rate",
    "dur",
    "gap_prev",
    "beat_resid",
    "onset_rate",
    "loud_rel",
    "loud_range",
    "loud_change_mag",
    "d_loud",
    "hnr_rel",
    "voiced_frac",
    "jitter_local",
    "chord_change",
    "chord_match",
    "tension",
    "tension_change",
    "phrase_f0",
    "phrase_loud",
    "pos_in_phrase",
)
PROSODY_V3_DIM = len(PROSODY_V3_FEATURES)


class ProsodyV3Extractor:
    def extract(self, vocal: Path, mix: Path, units: list[Unit]) -> np.ndarray:
        sound = parselmouth.Sound(str(vocal))
        audio, _ = librosa.load(str(mix), sr=SAMPLE_RATE, mono=True)
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )
        chroma_times = librosa.frames_to_time(
            np.arange(chroma.shape[1]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )
        onsets = librosa.onset.onset_detect(
            y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, units="time"
        )
        _, beats = librosa.beat.beat_track(
            y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, units="time"
        )
        beats = np.asarray(beats, dtype="float32")
        period = float(np.median(np.diff(beats))) if beats.size > 1 else 0.0
        tonic, _, _, _ = track_key(chroma)

        base = []
        for index, unit in enumerate(units):
            try:
                base.append(
                    self._word(
                        sound, chroma, chroma_times, onsets, beats, period, unit, tonic,
                        units[index - 1].end if index else unit.start,
                    )
                )
            except Exception as error:
                logger.debug("prosody-v3 failed for '%s': %s", unit.text, error)
                base.append(None)
        return _assemble(base, units)

    def _word(
        self, sound, chroma, times, onsets, beats, period, unit, tonic, prev_end
    ) -> dict:
        part = sound.extract_part(from_time=unit.start, to_time=unit.end)
        duration = max(float(unit.end - unit.start), 1e-6)
        pitch = part.to_pitch(
            time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING
        )
        frequency = pitch.selected_array["frequency"]
        stamps = pitch.xs()
        voiced = frequency > 0
        if voiced.sum() >= 2:
            f0 = frequency[voiced]
            semitones = 12.0 * np.log2(f0)
            rise = float((np.diff(semitones) > 0).mean())
            fall = float((np.diff(semitones) < 0).mean())
            stats = {
                "f0_hz": float(np.median(f0)),
                "f0_iqr": float(
                    np.percentile(semitones, 75) - np.percentile(semitones, 25)
                ),
                "f0_range": float(semitones.max() - semitones.min()),
                "f0_slope": float(np.polyfit(stamps[voiced], semitones, 1)[0]),
                "rise": rise,
                "fall": fall,
            }
        else:
            stats = {"f0_hz": 0.0, "f0_iqr": 0.0, "f0_range": 0.0,
                     "f0_slope": 0.0, "rise": 0.0, "fall": 0.0}
        stats["voiced"] = float(voiced.mean()) if voiced.size else 0.0

        intensity = part.to_intensity(time_step=TIME_STEP).values.flatten()
        if intensity.size:
            changes = np.abs(np.diff(intensity))
            stats["loud"] = float(intensity.mean())
            stats["loud_range"] = float(intensity.max() - intensity.min())
            stats["loud_mag"] = float(changes.mean()) if changes.size else 0.0
        else:
            stats.update({"loud": np.nan, "loud_range": 0.0, "loud_mag": 0.0})

        try:
            point = call(
                part, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING
            )
            stats["jitter"] = float(
                call(point, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            )
        except Exception:
            stats["jitter"] = np.nan
        try:
            harmonicity = call(
                part, "To Harmonicity (cc)", TIME_STEP, PITCH_FLOOR, 0.1, 1.0
            )
            stats["hnr"] = float(call(harmonicity, "Get mean", 0, 0))
        except Exception:
            stats["hnr"] = np.nan

        mid = (unit.start + unit.end) / 2
        selected = (times >= unit.start) & (times <= unit.end)
        if not selected.any():
            selected = np.zeros(chroma.shape[1], dtype=bool)
            selected[np.argmin(np.abs(times - mid))] = True
        vector = chroma[:, selected].mean(axis=1)
        normalized = vector / (vector.sum() + 1e-9)
        root, major, match, tension = chord_and_tension(np.roll(normalized, -tonic))
        stats.update({
            "chord": np.array([root, major]),
            "match": match,
            "tension": tension,
            "beat": beat_residual(mid, beats, period),
            "onset": float(
                np.sum((onsets >= unit.start) & (onsets <= unit.end)) / duration
            ),
            "dur": duration,
            "gap": max(float(unit.start - prev_end), 0.0),
            "rate": len(unit.text) / duration,
        })
        return stats


def _assemble(base: list, units: list[Unit]) -> np.ndarray:
    count = len(base)
    f0 = np.array([b["f0_hz"] if b else np.nan for b in base], dtype="float32")
    f0_rel = relative_st(f0)
    loud = np.array([b["loud"] if b else np.nan for b in base])
    loud_rel = loud - np.nanmean(loud)
    hnr = np.array([b["hnr"] if b else np.nan for b in base])
    hnr_rel = hnr - np.nanmean(hnr)
    tension = np.array([b["tension"] if b else np.nan for b in base])
    rows = []
    for index, stats in enumerate(base):
        if stats is None:
            rows.append([np.nan] * PROSODY_V3_DIM)
            continue
        prev = base[index - 1] if index else None
        d_f0 = f0_rel[index] - f0_rel[index - 1] if prev else 0.0
        d_loud = loud_rel[index] - loud_rel[index - 1] if prev else 0.0
        d_tension = tension[index] - tension[index - 1] if prev else 0.0
        chord_change = (
            float(np.abs(stats["chord"] - prev["chord"]).sum()) if prev else 0.0
        )
        phrase = _phrase(base, units, index)
        members = [f0_rel[i] for i in phrase]
        louds = [loud_rel[i] for i in phrase]
        rows.append([
            f0_rel[index],
            stats["f0_iqr"],
            stats["f0_range"],
            stats["f0_slope"],
            stats["rise"] - stats["fall"],
            stats["rise"] + stats["fall"],
            d_f0,
            stats["rate"],
            stats["dur"],
            stats["gap"],
            stats["beat"],
            stats["onset"],
            loud_rel[index],
            stats["loud_range"],
            stats["loud_mag"],
            d_loud,
            hnr_rel[index],
            stats["voiced"],
            stats["jitter"],
            chord_change,
            stats["match"],
            stats["tension"],
            d_tension,
            float(np.nanmean(members)),
            float(np.nanmean(louds)),
            phrase.index(index) / max(len(phrase) - 1, 1),
        ])
    matrix = np.array(rows, dtype="float32")
    names = list(PROSODY_V3_FEATURES)
    short = matrix[:, names.index("dur")] < MIN_DUR_S
    matrix[short, names.index("speech_rate")] = np.nan
    for name, (low, high) in CLIP.items():
        column = matrix[:, names.index(name)]
        matrix[:, names.index(name)] = np.where(
            np.isnan(column), np.nan, np.clip(column, low, high)
        )
    return matrix


def _phrase(base: list, units: list[Unit], index: int) -> list[int]:
    start = index
    while start > 0 and units[start].start - units[start - 1].end <= PHRASE_GAP_S:
        start -= 1
    end = index
    while (
        end + 1 < len(units)
        and units[end + 1].start - units[end].end <= PHRASE_GAP_S
    ):
        end += 1
    return list(range(start, end + 1))


def prosody_v3_dataset(
    config: ProjectConfig,
    paths: ProjectPaths,
    track_ids: Iterable[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    dataset = config.data.dataset
    vocal_dir = paths.vocals(dataset)
    raw_dir = paths.raw_dataset(dataset)
    target_dir = paths.features(dataset, "word", "prosody-v3")

    tracks = sorted(paths.transcriptions(dataset).glob("*.json"))
    if track_ids is not None:
        wanted = set(track_ids)
        tracks = [path for path in tracks if path.stem in wanted]
    if limit is not None:
        tracks = tracks[:limit]

    if not tracks:
        logger.warning("no transcriptions found for %s", dataset)
        return []

    extractor = ProsodyV3Extractor()
    done: list[str] = []
    skipped = 0
    for index, track in enumerate(tracks, start=1):
        track_id = track.stem
        target = target_dir / f"{track_id}.npz"
        if target.exists() and not overwrite:
            logger.debug("skipping existing %s", track_id)
            continue
        vocal = vocal_dir / f"{track_id}.wav"
        mix = find_audio(raw_dir, track_id)
        units = load_units(config, paths, track_id, "word")
        if not vocal.exists() or mix is None or not units:
            logger.warning("[%d/%d] %s has no inputs", index, len(tracks), track_id)
            skipped += 1
            continue
        try:
            features = extractor.extract(vocal, mix, units)
        except Exception as error:
            logger.error("[%d/%d] %s failed: %s", index, len(tracks), track_id, error)
            skipped += 1
            continue
        write_features(target, features, units)
        logger.info("[%d/%d] %s -> %d words", index, len(tracks), track_id, len(units))
        done.append(track_id)

    logger.info("wrote %d track(s) to %s (%d skipped)", len(done), target_dir, skipped)
    return done
