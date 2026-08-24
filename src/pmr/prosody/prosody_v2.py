import logging
from collections.abc import Iterable
from pathlib import Path

import librosa
import numpy as np
import parselmouth

from pmr.config import ProjectConfig
from pmr.features import Unit, find_audio, load_units, write_features
from pmr.paths import ProjectPaths
from pmr.prosody.common import (
    HOP_LENGTH,
    INTERVAL_EDGES,
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

PERFORMANCE_FEATURES = (
    "f0_median_rel_st",
    "f0_iqr",
    "f0_range_st",
    "f0_slope",
    "f0_rise_frac",
    "f0_fall_frac",
    "interval_bin0",
    "interval_bin1",
    "interval_bin2",
    "interval_bin3",
    "interval_bin4",
    "interval_bin5",
    "transition_updown",
    "transition_downup",
    "vibrato_rate",
    "vibrato_extent",
    "onset_rate",
    "beat_resid",
    "loud_mean_rel",
    "loud_std",
    "loud_range",
    "loud_change_count",
    "loud_change_mag",
    "attack_slope",
    "decay_slope",
    "gap_frac",
    "interval_mean",
    "interval_std",
)
TONALITY_FEATURES = (
    *(f"chroma_rel_{index}" for index in range(12)),
    "chord_root_rel",
    "chord_major",
    "chord_match",
    "tension",
    "tension_change",
    "chord_change",
)
GLOBAL_FEATURES = (
    *(f"key_{index}" for index in range(24)),
    "mode",
    "key_confidence",
    "tempo",
    "rubato",
    "tempo_var",
)
PROSODY_V2_FEATURES = PERFORMANCE_FEATURES + TONALITY_FEATURES
PROSODY_V2_DIM = len(PROSODY_V2_FEATURES)

CLIP = {
    "f0_median_rel_st": (-12.0, 12.0),
    "f0_iqr": (0.0, 12.0),
    "f0_range_st": (0.0, 12.0),
    "f0_slope": (-50.0, 50.0),
    "vibrato_extent": (0.0, 200.0),
    "onset_rate": (0.0, 10.0),
    "loud_mean_rel": (-30.0, 30.0),
    "loud_std": (0.0, 30.0),
    "loud_range": (0.0, 60.0),
    "loud_change_count": (0.0, 50.0),
    "loud_change_mag": (0.0, 10.0),
    "attack_slope": (-50.0, 50.0),
    "decay_slope": (-50.0, 50.0),
    "interval_mean": (-12.0, 12.0),
    "interval_std": (0.0, 12.0),
}
F0_ROW = 0
LOUD_ROW = PERFORMANCE_FEATURES.index("loud_mean_rel")


class ProsodyV2Extractor:
    def extract(
        self, vocal: Path, mix: Path, units: list[Unit]
    ) -> tuple[np.ndarray, np.ndarray]:
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
        tonic, mode, confidence, key_onehot = track_key(chroma)
        intervals = np.diff(onsets) if len(onsets) > 1 else np.array([0.0])
        tempo_var = float(intervals.std())
        beat = np.median(intervals)
        rubato = float(np.abs(intervals - beat).mean() / beat) if beat > 0 else 0.0
        tempo = 60.0 / period if period > 0 else 0.0

        rows, prev_tension, prev_chord = [], None, None
        for unit in units:
            try:
                performance, tonality, tension, chord = self._word(
                    sound, chroma, chroma_times, onsets, beats, period, unit, tonic
                )
                change = tension - prev_tension if prev_tension is not None else 0.0
                chord_change = (
                    float(np.abs(chord - prev_chord).sum())
                    if prev_chord is not None
                    else 0.0
                )
                prev_tension, prev_chord = tension, chord
                rows.append([*performance, *tonality, tension, change, chord_change])
            except Exception as error:
                logger.debug("prosody-v2 failed for '%s': %s", unit.text, error)
                rows.append([np.nan] * PROSODY_V2_DIM)
        words = np.array(rows, dtype="float32")
        words[:, F0_ROW] = relative_st(words[:, F0_ROW])
        words[:, LOUD_ROW] = words[:, LOUD_ROW] - np.nanmean(words[:, LOUD_ROW])
        names = list(PROSODY_V2_FEATURES)
        for name, (low, high) in CLIP.items():
            column = words[:, names.index(name)]
            words[:, names.index(name)] = np.where(
                np.isnan(column), np.nan, np.clip(column, low, high)
            )
        global_vector = np.concatenate(
            [key_onehot, [mode, confidence, tempo, rubato, tempo_var]]
        ).astype("float32")
        return words, global_vector

    def _word(
        self, sound, chroma, times, onsets, beats, period, unit, tonic
    ) -> tuple[list, list, float, np.ndarray]:
        part = sound.extract_part(from_time=unit.start, to_time=unit.end)
        duration = float(unit.end - unit.start)
        pitch = part.to_pitch(
            time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING
        )
        frequency = pitch.selected_array["frequency"]
        stamps = pitch.xs()
        voiced = frequency > 0
        if voiced.sum() >= 2:
            f0 = frequency[voiced]
            semitones = 12.0 * np.log2(f0)
            differences = np.diff(semitones)
            histogram, _ = np.histogram(np.abs(differences), bins=INTERVAL_EDGES)
            histogram = (histogram / max(len(differences), 1)).tolist()
            upward = float((differences > 0).mean())
            downward = float((differences < 0).mean())
            updown = (
                float(((differences[:-1] > 0) & (differences[1:] < 0)).mean())
                if len(differences) > 1
                else 0.0
            )
            downup = (
                float(((differences[:-1] < 0) & (differences[1:] > 0)).mean())
                if len(differences) > 1
                else 0.0
            )
            vibrato_rate, vibrato_extent = self._vibrato(stamps[voiced], semitones)
            pitch_stats = [
                float(np.median(f0)),
                float(np.percentile(f0, 75) - np.percentile(f0, 25)),
                float(semitones.max() - semitones.min()),
                float(np.polyfit(stamps[voiced], semitones, 1)[0]),
                upward,
                downward,
            ]
            contour = [float(differences.mean()), float(differences.std())]
        else:
            histogram = [0.0] * (len(INTERVAL_EDGES) - 1)
            updown = downup = vibrato_rate = vibrato_extent = 0.0
            pitch_stats = [0.0] * 6
            contour = [0.0, 0.0]
        voiced_fraction = float(voiced.mean()) if voiced.size else 0.0

        intensity = part.to_intensity(time_step=TIME_STEP).values.flatten()
        if intensity.size:
            changes = np.abs(np.diff(intensity))
            loud = [
                float(intensity.mean()),
                float(intensity.std()),
                float(intensity.max() - intensity.min()),
                float((changes > 3.0).sum()),
                float(changes.mean()),
                float(changes.max()),
                float(-changes.min()) if changes.size else 0.0,
            ]
        else:
            loud = [0.0] * 7

        onset_rate = float(
            np.sum((onsets >= unit.start) & (onsets <= unit.end)) / max(duration, 1e-6)
        )
        mid = (unit.start + unit.end) / 2
        performance = [
            *pitch_stats,
            *histogram,
            updown,
            downup,
            vibrato_rate,
            vibrato_extent,
            onset_rate,
            beat_residual(mid, beats, period),
            *loud,
            1.0 - voiced_fraction,
            *contour,
        ]

        selected = (times >= unit.start) & (times <= unit.end)
        if not selected.any():
            selected = np.zeros(chroma.shape[1], dtype=bool)
            selected[np.argmin(np.abs(times - mid))] = True
        vector = chroma[:, selected].mean(axis=1)
        normalized = vector / (vector.sum() + 1e-9)
        rotated = np.roll(normalized, -tonic)
        root, major, match, tension = chord_and_tension(rotated)
        tonality = [*rotated.tolist(), root, major, match]
        return performance, tonality, tension, np.asarray([root, major])

    def _vibrato(self, times, semitones):
        if len(times) < 5:
            return 0.0, 0.0
        grid = np.arange(times[0], times[-1], TIME_STEP)
        interpolated = np.interp(grid, times, semitones)
        interpolated = interpolated - interpolated.mean()
        spectrum = np.abs(np.fft.rfft(interpolated * np.hanning(len(interpolated))))
        frequencies = np.fft.rfftfreq(len(interpolated), d=TIME_STEP)
        band = (frequencies >= 3.0) & (frequencies <= 9.0)
        if not band.any():
            return 0.0, 0.0
        index = int(np.argmax(spectrum[band]))
        rate = float(frequencies[band][index])
        extent = float(spectrum[band][index] / max(len(interpolated), 1) * 1000.0)
        return rate, extent


def prosody_v2_dataset(
    config: ProjectConfig,
    paths: ProjectPaths,
    track_ids: Iterable[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    dataset = config.data.dataset
    vocal_dir = paths.vocals(dataset)
    raw_dir = paths.raw_dataset(dataset)
    target_dir = paths.features(dataset, "word", "prosody-v2")
    global_dir = paths.features(dataset, "track", "prosody-v2-global")

    tracks = sorted(paths.transcriptions(dataset).glob("*.json"))
    if track_ids is not None:
        wanted = set(track_ids)
        tracks = [path for path in tracks if path.stem in wanted]
    if limit is not None:
        tracks = tracks[:limit]

    if not tracks:
        logger.warning("no transcriptions found for %s", dataset)
        return []

    extractor = ProsodyV2Extractor()
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
            features, global_vector = extractor.extract(vocal, mix, units)
        except Exception as error:
            logger.error("[%d/%d] %s failed: %s", index, len(tracks), track_id, error)
            skipped += 1
            continue
        write_features(target, features, units)
        global_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            global_dir / f"{track_id}.npz",
            features=global_vector,
            names=np.array(GLOBAL_FEATURES),
        )
        logger.info("[%d/%d] %s -> %d words", index, len(tracks), track_id, len(units))
        done.append(track_id)

    logger.info("wrote %d track(s) to %s (%d skipped)", len(done), target_dir, skipped)
    return done
