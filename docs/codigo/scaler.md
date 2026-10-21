# `scaler.py` — Discretización y escalado conceptual

> **Archivo:** [`src/preprocessing/scaler.py`](../../../src/preprocessing/scaler.py) · **Fase:** 1 — Preparación de datos · **Teoría:** [§1.2 a §1.4](../../teoria.md#sec-1-2)

## Propósito

`scaler.py` es el punto de entrada de la Fase 1: convierte una tabla de pandas en la matriz de incidencia booleana de un contexto formal. Tiene dos funciones, que se aplican en este orden:

| Paso | Función | Entrada | Salida |
|:-:|---|---|---|
| 1 | [`discretize`](#discretize) | `DataFrame` con columnas numéricas y categóricas | El mismo `DataFrame`, con cada columna numérica reemplazada por códigos de intervalo $0, \dots, k - 1$ |
| 2 | [`scaling`](#scaling) | `DataFrame` con valores discretos | La matriz de incidencia: un `DataFrame` booleano de $\lvert G \rvert \times \lvert M \rvert$ |

`discretize` delega en los otros dos módulos de la carpeta: [`quantiles.py`](quantiles.md) decide cuántos intervalos usar y [`cuant_type.py`](cuant_type.md) decide dónde cortar.

Los ejemplos de este documento se ejecutan desde la raíz del repositorio, con `src/` como paquete de Python (la estructura del proyecto incluye los archivos `__init__.py`).

<a name="discretize"></a>

## `discretize(data, cuant_type, quantiles)`

Discretiza cada columna numérica de `data` y deja las demás sin cambios.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `data` | `pd.DataFrame` | Tabla original. Solo se transforman las columnas que pandas considera numéricas (`select_dtypes(include="number")`); las de texto y las booleanas se copian igual. |
| `cuant_type` | `str` | Método de corte: `"equal_width"`, `"equal_frequency"` o `"kmeans_1d"`. Ver [`cuant_type.md`](cuant_type.md). |
| `quantiles` | `int` o `str` | Número de intervalos: un entero $k \ge 2$, igual para todas las columnas, o el nombre de una regla (`"sturges"`, `"freedman"` o `"scott"`) que calcula $k$ por separado para cada columna. Ver [`quantiles.md`](quantiles.md). |

**Devuelve** una tupla `(out, ks)`:

- `out`: una copia de `data` en la que cada columna numérica se reemplazó por sus códigos de intervalo;
- `ks`: un diccionario `{columna: k}` con el número de intervalos usado en cada columna numérica.

**Errores:** `ValueError` si `quantiles` no es texto ni un entero mayor o igual que 2 (un `np.int64` también se rechaza, así que hay que convertirlo con `int(...)`), y `KeyError` si el método o la regla no existen.

```text
DISCRETIZAR(D, método, q)
Entrada: tabla D; nombre del método de corte; q = entero ≥ 2 o nombre de una regla
Salida:  tabla discretizada D' y diccionario K con el k de cada columna numérica

 1  si q no es texto y (q no es entero o q < 2) entonces error
 2  cortar ← MÉTODOS_DE_CORTE[método]                            ▹ cuant_type.cut_methods
 3  D' ← copia de D
 4  K ← diccionario vacío
 5  para cada columna numérica c de D:
 6      x ← D[c]
 7      si q es texto entonces k ← máx(2, REGLAS[q](x sin NaN))  ▹ quantiles.methods
 8      si no k ← q
 9      K[c] ← k
10      D'[c] ← cortar(x, k)
11  devolver (D', K)
```

**Sobre `ks`.** Guarda el $k$ pedido, o el que calculó la regla, no el número de intervalos que realmente se produjeron. Con `equal_frequency` y `kmeans_1d` puede haber menos intervalos, y con cualquier método algunos códigos pueden no aparecer. El número de códigos que quedaron en una columna se obtiene con `out[col].nunique()`.

**Complejidad:** la suma, sobre las columnas numéricas, del costo de la regla y del método de corte. Por ejemplo, con $p$ columnas numéricas, la regla de Sturges y `equal_frequency`, el costo es $O(p \cdot n \log n)$.

<a name="scaling"></a>

## `scaling(data, type, direction="le")`

Aplica la misma escala a todas las columnas de `data` y devuelve la matriz de incidencia.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `data` | `pd.DataFrame` | Tabla con valores discretos, normalmente la salida de `discretize`. |
| `type` | `str` | `"nominal"` u `"ordinal"`. |
| `direction` | `str` | Solo para la escala ordinal: `"le"` (creciente, atributos `col<=v`) o `"ge"` (decreciente, atributos `col>=v`). Por defecto, `"le"`. |

**Devuelve** un `DataFrame` booleano con el mismo índice que `data` y una columna por cada par (columna, valor observado):

| Escala | Atributo generado | Es verdadero cuando | Teoría |
|---|---|---|---|
| Nominal | `col=v` | el valor de la fila es `v` | Proposición 1.12 |
| Ordinal, `"le"` | `col<=v` | el valor es menor o igual que `v` | Proposición 1.13 |
| Ordinal, `"ge"` | `col>=v` | el valor es mayor o igual que `v` | Proposición 1.13 |

Solo los valores que aparecen en los datos generan atributos, así que la matriz tiene $\sum_c \lvert V_c \rvert$ columnas, donde $V_c$ es el conjunto de valores distintos de la columna $c$ (Corolario 1.17). Ninguna de las dos escalas pierde información (Proposición 1.16).

**Errores:** `ValueError` si `type` no es `"nominal"` ni `"ordinal"`, o si `direction` no es `"le"` ni `"ge"`.

```text
ESCALAR(D, tipo, dirección = "le")
Entrada: tabla D con valores discretos; tipo ∈ {"nominal", "ordinal"}; dirección ∈ {"le", "ge"}
Salida:  matriz de incidencia booleana con el mismo índice que D

 1  si tipo = "nominal" entonces
 2      para cada columna c y cada valor observado v de c:
 3          nuevo atributo "c=v", verdadero en las filas donde D[c] = v
 4  si no, si tipo = "ordinal" entonces
 5      op ← (≤ si dirección = "le";  ≥ si dirección = "ge")
 6      para cada columna c:
 7          V ← valores distintos de c sin NaN, en orden ascendente
 8          para cada v ∈ V:
 9              nuevo atributo "c op v", verdadero en las filas donde D[c] op v
10  si no error
11  devolver la tabla formada por todos los atributos nuevos  ▹ NaN op v es falso
```

- La escala nominal usa `pd.get_dummies(..., prefix_sep="=", dtype=bool)`. La ordinal compara cada valor con todos los niveles de su columna a la vez mediante *broadcasting* de NumPy: `x[:, None] <= levels[None, :]` produce en una sola operación el bloque de $n \times r$ de una columna con $r$ niveles.
- Una fila con NaN en una columna no recibe ningún atributo de esa columna (su bloque queda en `False`), con cualquiera de las dos escalas.
- Si una columna discretizada tiene NaN, sus códigos son flotantes y los atributos se llaman `radio=0.0` en lugar de `radio=0`.
- Con `"le"`, el atributo del valor máximo (`col<=máx`) es verdadero en todas las filas sin NaN; con `"ge"` pasa lo mismo con `col>=mín` (Corolario 1.14). Son columnas que no distinguen objetos y se pueden eliminar.
- La escala ordinal solo tiene sentido si los valores tienen un orden con significado, como los códigos de `discretize`. Aplicada a texto, usa el orden alfabético (por ejemplo, `diagnosis<=B` y `diagnosis<=M`), y si la columna mezcla texto y NaN falla con `TypeError`.

**Complejidad:** la escala nominal cuesta $O(n \cdot \lvert M \rvert)$, donde $\lvert M \rvert$ es el número de atributos generados. La ordinal cuesta $O(n \log n + n r_c)$ por cada columna $c$ con $r_c$ niveles: ordenar los valores y hacer las comparaciones. En ambos casos el resultado ocupa $n \cdot \lvert M \rvert$ bytes.

## Combinar escalas

`scaling` aplica la misma escala a todas las columnas que recibe. Para usar la escala ordinal en las variables discretizadas y la nominal en las categóricas, se llama una vez por grupo de columnas y se concatenan los resultados:

```python
K_num = scaling(disc[columnas_numericas], "ordinal", "le")
K_cat = scaling(disc[columnas_categoricas], "nominal")
K = pd.concat([K_num, K_cat], axis=1)
```

La escala interordinal (Observación 1.11), que permite describir intervalos, se obtiene de la misma forma, concatenando las salidas de `"le"` y de `"ge"` para las mismas columnas.

<a name="ejemplo"></a>

## Ejemplo completo

Es el ejemplo de la [§1.6 de la teoría](../../teoria.md#sec-1-6), ejecutado con el código:

```python
import pandas as pd
from src.preprocessing.scaler import discretize, scaling

df = pd.DataFrame(
    {"radio":       [12.0, 13.5, 20.1, 11.2, 17.8, 14.6],
     "textura":     [15.7, 21.3, 17.8, 25.0, 19.9, 14.1],
     "diagnostico": ["B", "B", "M", "B", "M", "B"]},
    index=["g1", "g2", "g3", "g4", "g5", "g6"],
)

disc, ks = discretize(df, "equal_width", 3)
print(disc)
#     radio  textura diagnostico
# g1      0        0           B
# g2      0        1           B
# g3      2        1           M
# g4      0        2           B
# g5      2        1           M
# g6      1        0           B
print(ks)
# {'radio': 3, 'textura': 3}

K = pd.concat(
    [scaling(disc[["radio", "textura"]], "ordinal", "le"),
     scaling(disc[["diagnostico"]], "nominal")],
    axis=1,
)
print(K.astype(int).to_string())
#     radio<=0  radio<=1  radio<=2  textura<=0  textura<=1  textura<=2  diagnostico=B  diagnostico=M
# g1         1         1         1           1           1           1              1              0
# g2         1         1         1           0           1           1              1              0
# g3         0         0         1           0           1           1              0              1
# g4         1         1         1           0           0           1              1              0
# g5         0         0         1           0           1           1              0              1
# g6         0         1         1           1           1           1              1              0
```

`K` es la matriz de incidencia de un contexto con $\lvert G \rvert = 6$ objetos y $\lvert M \rvert = 8$ atributos. La [§1.6](../../teoria.md#sec-1-6) explica cómo leerla.

## Antes de discretizar: qué columnas pasar

`discretize` transforma todas las columnas numéricas sin distinguir qué significan, así que conviene preparar la tabla antes:

1. **Identificadores.** Una columna numérica como `id` se discretizaría como si fuera una medida. Hay que quitarla.
2. **Columnas vacías.** El CSV de Breast Cancer Wisconsin que se distribuye en Kaggle trae una columna completamente vacía, `Unnamed: 32`. Con ella, `discretize` falla con la regla `sturges` o con el método `equal_width`, y con los demás la columna queda toda en NaN o toda en 0. Hay que quitarla.
3. **La clase.** El diagnóstico (`M`/`B`) es texto, así que `discretize` no lo toca. Si se incluye en el contexto, aparece como atributo y entra en los conceptos. En la fase de Machine Learning no debe estar en el contexto con el que se construyen las características, porque eso le filtraría la respuesta al modelo.
4. **Atributos binarios codificados como 0 y 1.** Si un dataset, como el de especies animales, guarda atributos binarios como enteros 0/1, pandas los considera numéricos y `discretize` los partiría en intervalos. Lo natural es convertirlos a `bool` y usarlos directamente como columnas de la matriz de incidencia. Si se pasan por `scaling(..., "nominal")`, cada uno genera dos atributos, `col=False` y `col=True`, y normalmente basta con el segundo.
5. **Variables enteras con pocos valores** (por ejemplo, un número de patas). No necesitan discretización: se pueden escalar directamente con `scaling`.

## Casos borde

| Situación | Comportamiento |
|---|---|
| Columna numérica completamente vacía | `ValueError` con la regla `sturges` o con `equal_width`; con los otros métodos queda toda en NaN (`equal_frequency`) o toda en 0 (`kmeans_1d`) |
| Columna constante con `equal_frequency` | Queda toda en NaN y no genera atributos al escalar |
| NaN en una columna | La fila no recibe atributos de esa columna, y los nombres pasan a ser del tipo `col=0.0` |
| `quantiles` es un `np.int64` | `ValueError`; hay que convertirlo con `int(...)` |
| Método o regla inexistente | `KeyError` |
| Menos intervalos de los pedidos | `ks` sigue mostrando el $k$ pedido |
| Escala ordinal sobre texto | Usa el orden alfabético; si hay texto y NaN mezclados, `TypeError` |
| Escala `"le"` sin NaN | La columna `col<=máx` está llena de `True` |

Los casos borde de cada método de corte están en [`cuant_type.md`](cuant_type.md#casos-borde), y los de cada regla en [`quantiles.md`](quantiles.md#casos-borde).
