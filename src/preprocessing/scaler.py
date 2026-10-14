import pandas as pd
from quantiles import methods
from cuant_type import cut_methods

def discretize(data: pd.DataFrame, cuant_type: str, quantiles: str) -> pd.DataFrame:
    df_result = data.copy()
    numeric_cols = df_result.select_dtypes(include='number').columns

    for col in numeric_cols:
        k = methods[quantiles](df_result[col])
        df_result[col] = cut_methods[cuant_type](df_result[col], k)

    return df_result
