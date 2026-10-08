from typing import Callable, Dict
import numpy as np
import pandas as pd
from numpy.typing import NDArray

CutFunc = Callable[[pd.Series, int], pd.Series]

cut_methods: Dict[str, CutFunc] = {}

def __method(name: str) -> Callable[[CutFunc], CutFunc]:
    def decorador(func: CutFunc) -> CutFunc:
        cut_methods[name] = func
        return func
    return decorador

@__method("equal_width")
def equal_width(serie: pd.Series, k: int) -> pd.Series:
    return pd.cut(serie, bins=k, labels=False, include_lowest=True)

@__method("equal_frequency")
def equal_frequency(serie: pd.Series, k: int) -> pd.Series:
    return pd.qcut(serie, q=k, labels=False, duplicates='drop')

@__method("kmeans_1d")
def kmeans_1d(serie: pd.Series, k: int) -> pd.Series:
    vals: NDArray[np.float64] = np.sort(serie.dropna().to_numpy(dtype=np.float64))

    if len(vals) <= k or vals.min() == vals.max():

        return pd.Series(0, index=serie.index, dtype=np.int64)


    percentiles: NDArray[np.float64] = np.linspace(0.0, 100.0, k + 2, dtype=np.float64)[1:-1]
    centroids: NDArray[np.float64] = np.percentile(vals, percentiles)

    for _ in range(30):
        distances: NDArray[np.float64] = np.abs(vals[:, None] - centroids[None, :])
        labels: NDArray[np.int64] = np.argmin(distances, axis=1)

        new_centroids: NDArray[np.float64] = np.array(
            [vals[labels == i].mean() if np.any(labels == i) else centroids[i] for i in range(k)],
            dtype=np.float64,
        )

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    centroids = np.sort(centroids)
    midpoints: NDArray[np.float64] = (centroids[:-1] + centroids[1:]) / 2.0

    cuts: NDArray[np.float64] = np.unique(np.concatenate(([-np.inf], midpoints, [np.inf])))

    return pd.cut(serie, bins=cuts, labels=False, include_lowest=True)
