from math import ceil, log2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, Callable

methods: Dict[str, Callable[[pd.Series], int]] = {}
def __methods(name: str) -> Callable[[pd.Series], int]:
    def decorador(func: Callable[[pd.Series, int], int]) -> Callable[[pd.Series, int], int]:
        methods[name] = func

        return func

    return decorador

@__methods("sturges")
def sturges(serie: pd.Series) -> int:
    return ceil(log2(len(serie)) + 1)

@__methods("freedman")
def freedman(serie: pd.Series) -> int:
    vals: NDArray[np.float64] = serie.to_numpy(dtype=np.float64)
    n: int = len(vals)
    if n <= 1:
        return 1

    q25: float
    q75: float
    q25, q75 = map(float, np.percentile(vals, [25, 75]))

    iqr: float = q75 - q25
    rango: float = float(vals.max() - vals.min())

    if iqr == 0.0 or rango == 0.0:
        return sturges(serie)

    h: float = (2.0 * iqr) / (n ** (1.0 / 3.0))
    return ceil(rango / h)

@__methods("scott")
def scott(serie: pd.Series) -> int:
    vals: NDArray[np.float64] = serie.to_numpy(dtype=np.float64)
    n: int = len(vals)
    if n <= 1:
        return 1

    std: float = float(vals.std(ddof=1))
    rango: float = float(vals.max() - vals.min())

    if std == 0.0 or rango == 0.0:
        return sturges(serie)

    h: float = (3.49 * std) / (n ** (1.0 / 3.0))
    return ceil(rango / h)
