import glob
import os
import shutil
import kagglehub
import pandas as pd
from preprocessing.scaler import discretize, scaling

import random
import time
from itertools import islice

from fca.context import FormalContext, bits

path = kagglehub.dataset_download("kennyloic/animal-species-data")

csv_file = glob.glob(os.path.join(path, "*.csv"))[0]
data = pd.read_csv(csv_file)

shutil.rmtree(path, ignore_errors=True)

print("--- DataFrame Original (Primeras 5 filas) ---")
print(data.head())
print("\n" + "=" * 50 + "\n")

df_ew_sturges = discretize(data, cuant_type="kmeans_1d", quantiles=2)[0]
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


fc = FormalContext(matriz_final)
print(f"\nObjetos: {fc.n_obj} | Atributos: {fc.n_att}")

M = matriz_final.to_numpy(dtype=bool)
for g in range(fc.n_obj):
    for j in range(fc.n_att):
        assert (fc.itx[g] >> j & 1) == M[g, j]
        assert (fc.ext[j] >> g & 1) == M[g, j]
print("OK: ext e itx coinciden con la matriz")

rng = random.Random(0)

def rand_mask(p: float = 0.05) -> int:
    return sum(1 << j for j in range(fc.n_att) if rng.random() < p)

for _ in range(1000):
    B = rand_mask()
    C = fc.closure(B)
    assert B & C == B                                  # B ⊆ B''
    assert fc.closure(C) == C                          # idempotencia
    assert fc.extent(B) == fc.extent(C)                # B' = B'''
    C2 = fc.closure(B | rand_mask())
    assert C & C2 == C                                 # monotonía
print("OK: propiedades del cierre")

def all_intents(fc: FormalContext):
    A = fc.closure(0)
    while A is not None:
        yield A
        A = fc.next_closure(A)

def lectic_key(mask: int, n: int) -> int:
    return int(format(mask, f"0{n}b")[::-1], 2)

LIMIT = 1000
t0 = time.perf_counter()
intents = list(islice(all_intents(fc), LIMIT))
dt = time.perf_counter() - t0

keys = [lectic_key(a, fc.n_att) for a in intents]
assert all(fc.closure(a) == a for a in intents)
assert all(k1 < k2 for k1, k2 in zip(keys, keys[1:]))
print(f"OK: {len(intents)} intents en {dt:.3f}s ({dt/len(intents)*1000:.2f} ms c/u)")

for a in intents[:5]:
    ext_names = fc.extent(a)
    print(f"\nintent : {fc._to_names(a, fc.attributes)}")
    print(f"extent : {len(fc._to_names(ext_names, fc.objects))} objetos")

sub  = matriz_final.sample(n=12, axis=1, random_state=0)
fc_s = FormalContext(sub)

lista  = list(all_intents(fc_s))
brute  = {s for s in range(1 << fc_s.n_att) if fc_s.closure(s) == s}

assert len(lista) == len(set(lista))
assert set(lista) == brute
print(f"OK: {len(lista)} intents, coinciden con fuerza bruta")

t0, n = time.perf_counter(), 0
for _ in all_intents(fc):
    n += 1
    if time.perf_counter() - t0 > 30:
        print(f"Sin terminar: {n} conceptos en 30 s")
        break
else:
    print(f"Total: {n} conceptos en {time.perf_counter() - t0:.1f} s")
