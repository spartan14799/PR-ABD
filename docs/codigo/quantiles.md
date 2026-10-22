# `quantiles.py` — Reglas para el número de intervalos

> **Archivo:** [`src/preprocessing/quantiles.py`](../../../src/preprocessing/quantiles.py) · **Fase:** 1 — Preparación de datos · **Teoría:** §1.5 de los [fundamentos teóricos](../../teoria/teoria.pdf)

## Propósito

Antes de discretizar una variable numérica hay que decidir en cuántos intervalos dividirla. Este módulo implementa tres reglas clásicas que calculan ese número $k$ a partir de los datos de una columna. La función [`discretize`](scaler.md#discretize) de `scaler.py` las usa cuando su parámetro `quantiles` es el nombre de una regla en lugar de un entero.

Pese al nombre del archivo, las funciones no devuelven cuantiles sino un número de intervalos.

## Contenido del módulo

| Nombre | Firma | Devuelve |
|---|---|---|
| `methods` | `Dict[str, Callable[[pd.Series], int]]` | Registro de las reglas por nombre: `"sturges"`, `"freedman"` y `"scott"` |
| `sturges` | `(serie: pd.Series) -> int` | $\lceil \log_2 n \rceil + 1$ |
| `freedman` | `(serie: pd.Series) -> int` | $\lceil R/h \rceil$, con $h = 2 \cdot \mathrm{IQR} \cdot n^{-1/3}$ |
| `scott` | `(serie: pd.Series) -> int` | $\lceil R/h \rceil$, con $h = 3.49 \cdot s \cdot n^{-1/3}$ |

Aquí $n$ es el número de valores, $R = \max - \min$ es el rango, $\mathrm{IQR}$ es el rango intercuartílico y $s$ es la desviación estándar muestral. Las tres funciones suponen que la serie no tiene NaN; `discretize` se encarga de eso pasándoles `serie.dropna()`.

## Registro de reglas

El módulo usa un decorador para registrar cada regla en el diccionario `methods`:

```python
methods: Dict[str, Callable[[pd.Series], int]] = {}

def __methods(name):
    def decorador(func):
        methods[name] = func      # guarda la función con su nombre
        return func
    return decorador

@__methods("sturges")
def sturges(serie): ...
```

Como `discretize` solo consulta `methods[nombre]`, agregar una regla no exige tocar `scaler.py`. Por ejemplo, la regla de Rice, $k = \lceil 2 n^{1/3} \rceil$, se agregaría en `quantiles.py` así:

```python
@__methods("rice")
def rice(serie: pd.Series) -> int:
    return ceil(2 * len(serie) ** (1 / 3))
```

y quedaría disponible como `discretize(df, "equal_width", "rice")`.

## Funciones

### `sturges(serie)`

Regla de Sturges (Regla 1.29 de la teoría): $k = \lceil \log_2 n + 1 \rceil$, que es igual a $\lceil \log_2 n \rceil + 1$. Solo depende del número de valores, no de su dispersión. Con $n = 101$ da 8 y con $n = 569$ da 11.

```text
STURGES(x)
Entrada: serie x con n valores, sin NaN
Salida:  número de intervalos k

1  n ← |x|
2  devolver ⌈log₂(n) + 1⌉
```

### `freedman(serie)`

Regla de Freedman–Diaconis (Regla 1.33): cada intervalo mide $h = 2 \cdot \mathrm{IQR} \cdot n^{-1/3}$, y el número de intervalos es el necesario para cubrir el rango.

```text
FREEDMAN(x)
Entrada: serie x con n valores reales, sin NaN
Salida:  número de intervalos k

 1  n ← |x|
 2  si n ≤ 1 entonces devolver 1
 3  q₁ ← percentil(x, 25)
 4  q₃ ← percentil(x, 75)
 5  IQR ← q₃ − q₁
 6  R ← máx(x) − mín(x)
 7  si IQR = 0 o R = 0 entonces  ▹ el ancho h sería 0
 8      devolver STURGES(x)
 9  h ← 2 · IQR / ∛n
10  devolver ⌈R / h⌉
```

- Los percentiles se calculan con interpolación lineal, la misma definición de cuantil de la Definición 1.21.
- $\mathrm{IQR} = 0$ ocurre cuando la mitad central de los datos es constante, por ejemplo en una variable con muchos ceros.
- Con valores atípicos el rango $R$ crece pero el IQR no, así que $k$ puede salir muy grande.

### `scott(serie)`

Regla de Scott (Corolario 1.32): cada intervalo mide $h = 3.49 \cdot s \cdot n^{-1/3}$, que es el ancho óptimo cuando los datos son aproximadamente normales.

```text
SCOTT(x)
Entrada: serie x con n valores reales, sin NaN
Salida:  número de intervalos k

1  n ← |x|
2  si n ≤ 1 entonces devolver 1
3  s ← desviación estándar muestral de x  ▹ divisor n − 1
4  R ← máx(x) − mín(x)
5  si s = 0 o R = 0 entonces devolver STURGES(x)
6  h ← 3.49 · s / ∛n
7  devolver ⌈R / h⌉
```

Las condiciones $s = 0$ y $R = 0$ son equivalentes (ambas dicen que la variable es constante); comprobar las dos es solo una precaución.

## Comparación

| Regla | Depende de | Ventaja | Desventaja |
|---|---|---|---|
| Sturges | solo $n$ | Simple y estable | Ignora la dispersión y la forma; para $n$ grande produce pocos intervalos |
| Freedman–Diaconis | $n$, IQR y rango | Robusta: el IQR no se ve afectado por valores atípicos | Con atípicos el rango crece y $k$ puede ser muy grande |
| Scott | $n$, $s$ y rango | Óptima (en el sentido AMISE) para datos aproximadamente normales | $s$ es sensible a los valores atípicos |

Las tres reglas se diseñaron para histogramas de igual ancho. Usarlas para elegir $k$ con igual frecuencia o con k-medias es una heurística razonable, no un resultado de optimalidad (Observación 1.34).

## Complejidad

| Función | Tiempo | Memoria adicional |
|---|---|---|
| `sturges` | $O(1)$ | $O(1)$ |
| `freedman` | $O(n)$: percentiles por selección, mínimo y máximo | $O(n)$: copia a un arreglo de NumPy |
| `scott` | $O(n)$ | $O(n)$ |

## Casos borde

| Situación | `sturges` | `freedman` | `scott` |
|---|---|---|---|
| Serie vacía ($n = 0$) | `ValueError: math domain error` | Devuelve 1 | Devuelve 1 |
| Un solo valor ($n = 1$) | Devuelve 1 | Devuelve 1 | Devuelve 1 |
| Variable constante | Normal | Usa Sturges | Usa Sturges |
| Mitad central constante ($\mathrm{IQR} = 0$) | Normal | Usa Sturges | Normal |
| La serie tiene NaN | Cuenta los NaN en $n$ | `ValueError` | `ValueError` |

`discretize` evita el problema de los NaN pasando `serie.dropna()` y obliga a que $k \ge 2$. Sin embargo, una columna completamente vacía llega como una serie vacía y `sturges` falla (ver los [casos borde de `scaler.py`](scaler.md#casos-borde)).

## Ejemplo

```python
import pandas as pd
from src.preprocessing.quantiles import methods

radio = pd.Series([12.0, 13.5, 20.1, 11.2, 17.8, 14.6])
print({nombre: regla(radio) for nombre, regla in methods.items()})
# {'sturges': 4, 'freedman': 2, 'scott': 2}
```

Con $n = 6$, Sturges da $\lceil \log_2 6 + 1 \rceil = \lceil 3.58 \rceil = 4$. Para Freedman–Diaconis, $\mathrm{IQR} = 17.0 - 12.375 = 4.625$, luego $h \approx 5.09$ y $\lceil 8.9 / 5.09 \rceil = 2$.
