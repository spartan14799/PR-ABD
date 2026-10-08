from math import ceil, log2
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def sturges(serie: pd.Series) -> int:
    return ceil(log2(len(serie)) + 1)


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

def discretize(data: pd.DataFrame, cuant_type: str, quantiles: str) -> pd.DataFrame:
    # Solo valores numéricos son los que se discretizan
    df_numerico = data.select_dtypes(include='number')
    return None
