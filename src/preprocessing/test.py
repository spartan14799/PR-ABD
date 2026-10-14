import numpy as np
import pandas as pd
from scaler import discretize, scaling

np.random.seed(42)

data = pd.DataFrame(
    {
        "id": [f"ID_{i}" for i in range(100)],
        "edad": np.random.randint(18, 70, size=100),
        "ingresos": np.random.exponential(scale=2000, size=100)
        + 1000,
        "puntaje": np.random.normal(loc=50, scale=15, size=100),
    }
)

print("--- DataFrame Original (Primeras 5 filas) ---")
print(data.head())
print("\n" + "=" * 50 + "\n")

df_ew_sturges = discretize(data, cuant_type="equal_width", quantiles="sturges")[0]
print("--- Discretizado: equal_width + sturges ---")
print(df_ew_sturges.head())
print("\nConteo de clases en 'edad':")
print(df_ew_sturges["edad"].value_counts().sort_index())
print("\n" + "=" * 50 + "\n")

print(scaling(df_ew_sturges, "nominal"))
