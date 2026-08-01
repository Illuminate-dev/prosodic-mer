import numpy as np


def rmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(((prediction - target) ** 2).mean(axis=0))


def r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = ((target - prediction) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - residual / np.maximum(total, 1e-12)


def pcc(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = prediction - prediction.mean(axis=0)
    t = target - target.mean(axis=0)
    denominator = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
    return (p * t).sum(axis=0) / np.maximum(denominator, 1e-12)


def ccc(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = prediction - prediction.mean(axis=0)
    t = target - target.mean(axis=0)
    covariance = (p * t).mean(axis=0)
    variance = (p ** 2).mean(axis=0) + (t ** 2).mean(axis=0)
    bias = (prediction.mean(axis=0) - target.mean(axis=0)) ** 2
    return 2.0 * covariance / np.maximum(variance + bias, 1e-12)


def evaluate(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return {
        name: float(function(prediction, target).mean())
        for name, function in (
            ("rmse", rmse),
            ("r2", r2),
            ("pcc", pcc),
            ("ccc", ccc),
        )
    }
