import pandas as pd
from quantiles import methods
from cuant_type import cut_methods

def discretize(data: pd.DataFrame, cuant_type: str, quantiles: str) -> pd.DataFrame:
    # Solo valores numéricos son los que se discretizan
    df_numerico = data.select_dtypes(include='number')

    for row in df_numerico.columns:
        k = methods[quantiles](df_numerico[row])
        df_numerico[row] = cut_methods[cuant_type](df_numerico[row], k)
    return df_numerico
