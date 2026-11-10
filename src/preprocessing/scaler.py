import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union

from .quantiles import methods
from .cuant_type import cut_methods

def discretize(data: pd.DataFrame, cuant_type: str, quantiles: Union[str, int]) -> Tuple[pd.DataFrame, Dict[str, int]]:

    auto = isinstance(quantiles, str)
    if not auto and not (isinstance(quantiles, int) and quantiles >= 2):
        raise ValueError("Dato no válido para partir la información")

    cut = cut_methods[cuant_type]
    out = data.copy()
    ks: Dict[str, int] = {}

    for col in out.select_dtypes(include="number").columns:
        s = out[col]
        k = max(2, methods[quantiles](s.dropna())) if auto else quantiles
        ks[col] = k
        out[col] = cut(s, k)

    return out, ks


def scaling(data: pd.DataFrame, type: str, direction: str = "le") -> pd.DataFrame:
    if type == "nominal":
        return pd.get_dummies(data, columns=data.columns, prefix_sep="=", dtype=bool)

    if type == "ordinal":
        if direction not in ("le", "ge"):
            raise ValueError("direction debe ser 'le' o 'ge'")
        op, sym = (np.less_equal, "<=") if direction == "le" else (np.greater_equal, ">=")

        blocks, names = [], []
        for col in data.columns:
            x = data[col].to_numpy()
            levels = np.unique(x[~pd.isna(x)])
            blocks.append(op(x[:, None], levels[None, :]))
            names += [f"{col}{sym}{v}" for v in levels]

        return pd.DataFrame(np.hstack(blocks), index=data.index, columns=names)

    raise ValueError(f"Tipo no soportado: {type}")
