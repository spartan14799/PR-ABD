import glob
import os
import shutil
import kagglehub
import pandas as pd
from scaler import discretize, scaling

path = kagglehub.dataset_download("kennyloic/animal-species-data")

csv_file = glob.glob(os.path.join(path, "*.csv"))[0]
data = pd.read_csv(csv_file)

shutil.rmtree(path, ignore_errors=True)

print("--- DataFrame Original (Primeras 5 filas) ---")
print(data.head())
print("\n" + "=" * 50 + "\n")

df_ew_sturges = discretize(data, cuant_type="equal_width", quantiles="sturges")[0]
print("--- Discretizado: equal_width + sturges ---")
print(df_ew_sturges.head())

col_numerica = data.select_dtypes(include=["number"]).columns[0]
print(f"\nConteo de clases en '{col_numerica}':")
print(df_ew_sturges[col_numerica].value_counts().sort_index())
print("\n" + "=" * 50 + "\n")

matriz_final = scaling(df_ew_sturges, "nominal")

print("--- Matriz Final Escalada ---")
print(matriz_final)

print("\n--- Shape de la Matriz Final ---")
print("Dimensiones (filas, columnas):", matriz_final.shape)
