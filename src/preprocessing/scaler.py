import pandas as pd
from quantiles import methods
from cuant_type import cut_methods
from typing import Union

def discretize(data: pd.DataFrame, cuant_type: str, quantiles: Union[str, int]) -> pd.DataFrame:
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

    return df_result
