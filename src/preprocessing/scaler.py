import pandas as pd
import numpy as np
from typing import Union

from quantiles import methods
from cuant_type import cut_methods

def discretize(data: pd.DataFrame, cuant_type: str, quantiles: Union[str, int]) -> tuple[pd.DataFrame, int]:
    df_result = data.copy()
    numeric_cols = df_result.select_dtypes(include='number').columns

    for col in numeric_cols:
        if isinstance(quantiles, str):
            k = methods[quantiles](df_result[col])
        elif isinstance(quantiles, int) and quantiles >= 2:
            k = quantiles
        else:
            raise ValueError("Dato no válido para partir la información")

        df_result[col] = cut_methods[cuant_type](df_result[col], k)

    return df_result, k

def scaling(data: pd.DataFrame, type: str) -> np.ndarray:
    if type == "nominal":
        return pd.get_dummies(data, columns=data.columns, dtype=np.uint8).to_numpy()

    if type == "ordinal":
        numeric_data = data.select_dtypes(include=np.number)
        return numeric_data.apply(lambda col: col.astype('category').cat.codes).to_numpy()

    raise ValueError(f"Tipo no soportado: {type}")

