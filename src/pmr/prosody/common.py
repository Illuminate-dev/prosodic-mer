import numpy as np

PITCH_FLOOR = 75.0
PITCH_CEILING = 600.0
TIME_STEP = 0.01
SAMPLE_RATE = 22050
HOP_LENGTH = 512
INTERVAL_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0])
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
MAJOR_TEMPLATE = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
MINOR_TEMPLATE = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


def track_key(chroma: np.ndarray) -> tuple[int, float, float, np.ndarray]:
    profile = chroma.mean(axis=1)
    profile = profile / (profile.sum() + 1e-9)
    scores = []
    for index in range(12):
        scores.append(np.corrcoef(np.roll(MAJOR_PROFILE, index), profile)[0, 1])
        scores.append(np.corrcoef(np.roll(MINOR_PROFILE, index), profile)[0, 1])
    best = int(np.nanargmax(scores))
    key_onehot = np.zeros(24)
    key_onehot[best] = 1.0
    return best // 2, float(best % 2), float(scores[best]), key_onehot


def chord_and_tension(rotated: np.ndarray) -> tuple[float, float, float, float]:
    scores = []
    for index in range(12):
        scores.append(float(rotated @ np.roll(MAJOR_TEMPLATE, index)))
        scores.append(float(rotated @ np.roll(MINOR_TEMPLATE, index)))
    best = int(np.argmax(scores))
    tension = float(-(rotated * np.log(rotated + 1e-9)).sum())
    return float(best // 2), float(best % 2 == 0), float(scores[best]), tension


def beat_residual(mid: float, beats: np.ndarray, period: float) -> float:
    if beats.size < 2 or period <= 0:
        return 0.0
    return float(np.minimum(np.abs(mid - beats).min() / period, 0.5))


def relative_st(hz: np.ndarray) -> np.ndarray:
    out = np.where(np.isnan(hz), np.nan, 0.0).astype("float32")
    voiced = (hz > 0) & ~np.isnan(hz)
    if not voiced.any():
        return out
    track = np.median(hz[voiced])
    out[voiced] = 12.0 * np.log2(hz[voiced] / max(track, 1e-9))
    return out
