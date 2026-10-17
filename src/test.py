import glob
import os
import kagglehub
import pandas as pd

from preprocessing.scaler import discretize, scaling
from fca.context import FormalContext
from implications.guigues_duquenne import guigues_duquenne, format_implications

path = kagglehub.dataset_download("kennyloic/animal-species-data")
csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
df = pd.read_csv(csvs[0])
df = df.set_index("Animal Name")

df = df.drop(columns=["Animal Type", "Continent", "Country"])
print(df)
I = discretize(df, "kmeans_1d", 2)[0]
print(I)
I = scaling(I, "nominal")
print(I)
fc = FormalContext(I)
L = guigues_duquenne(fc)

print(f"Atributos resultantes: {len(I.columns)}")
print(f"Total de implicaciones en la Base Canónica: {len(L)}")

for s in format_implications(L, fc)[:10]:
     print(s)
