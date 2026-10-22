# `cuant_type.py` — Métodos de discretización

> **Archivo:** [`src/preprocessing/cuant_type.py`](../../../src/preprocessing/cuant_type.py) · **Fase:** 1 — Preparación de datos · **Teoría:** §1.4 de los [fundamentos teóricos](../../teoria/teoria.pdf)

## Propósito

Este módulo convierte una columna numérica en una columna de códigos de intervalo $0, 1, \dots, k - 1$. Implementa tres formas de elegir los cortes, registradas en el diccionario `cut_methods`, que usa [`discretize`](scaler.md#discretize) en `scaler.py`.

## Contenido del módulo

| Nombre | Firma | Descripción |
|---|---|---|
| `CutFunc` | `Callable[[pd.Series, int], pd.Series]` | Tipo común de los métodos de corte |
| `cut_methods` | `Dict[str, CutFunc]` | Registro por nombre: `"equal_width"`, `"equal_frequency"` y `"kmeans_1d"` |
| `equal_width` | `(serie, k) -> pd.Series` | Intervalos de igual ancho |
| `equal_frequency` | `(serie, k) -> pd.Series` | Intervalos con aproximadamente el mismo número de objetos |
| `kmeans_1d` | `(serie, k) -> pd.Series` | Intervalos alrededor de los centroides de k-medias |

El registro funciona igual que en [`quantiles.py`](quantiles.md#registro-de-reglas): un decorador agrega cada función al diccionario, de modo que se pueden añadir métodos nuevos sin tocar `scaler.py`.

**Contrato común de los tres métodos**

- *Entrada:* una serie numérica, que puede tener NaN, y un entero $k \ge 2$.
- *Salida:* una serie con el mismo índice y el código del intervalo de cada valor. Los códigos son enteros (`int64`), o flotantes (`float64`) si hay NaN, porque pandas representa así los faltantes.
- *Orden:* los tres métodos conservan el orden: si $x \le y$, el código de $x$ es menor o igual que el de $y$ (Proposición 1.19). Por eso tiene sentido aplicar después una escala ordinal.
- *Número de intervalos:* puede haber menos de $k$ intervalos efectivos (por cortes repetidos o centroides que coinciden), y algunos códigos pueden no aparecer. Ver los [casos borde](#casos-borde).

## Funciones

### `equal_width(serie, k)`

Divide el rango $[\min x, \max x]$ en $k$ intervalos de ancho $(\max x - \min x)/k$ (Definición 1.20). Usa `pd.cut(serie, bins=k, labels=False, include_lowest=True)`.

```text
ANCHO_IGUAL(x, k)
Entrada: serie x (puede tener NaN), entero k ≥ 2
Salida:  código y_g ∈ {0, …, k−1} para cada fila g (NaN si x_g es NaN)

1  a ← mín(x);  b ← máx(x)      ▹ ignorando los NaN
2  para j ← 0, …, k:  c_j ← a + j · (b − a) / k
3  c_0 ← c_0 − 0.001 · (b − a)  ▹ pandas baja el primer corte para incluir a
4  para cada fila g:
5      si x_g es NaN entonces y_g ← NaN
6      si no y_g ← el j tal que c_j < x_g ≤ c_{j+1}
7  devolver y
```

- Los intervalos son cerrados por la derecha, $(c_j, c_{j+1}]$. Como pandas baja el primer corte un 0.1 % del rango, el mínimo queda dentro del primer intervalo, y `include_lowest=True` no tiene efecto práctico.
- Si la variable es constante, pandas ensancha el rango un 0.1 % a cada lado y todos los valores reciben el mismo código (1 con $k = 3$).
- El número de objetos por intervalo puede ser muy desigual, y con valores atípicos casi todos pueden caer en el mismo intervalo.

**Complejidad:** $O(n \log k)$: el mínimo y el máximo en $O(n)$, y una búsqueda binaria entre los $k + 1$ cortes para cada valor.

### `equal_frequency(serie, k)`

Pone los cortes en los cuantiles empíricos de orden $0, 1/k, 2/k, \dots, 1$ (Definición 1.21), de modo que cada intervalo tenga aproximadamente $n/k$ objetos. Con $k = 4$, los cortes interiores son los cuartiles. Usa `pd.qcut(serie, q=k, labels=False, duplicates="drop")`.

```text
FRECUENCIA_IGUAL(x, k)
Entrada: serie x (puede tener NaN), entero k ≥ 2
Salida:  código y_g para cada fila g (NaN si x_g es NaN)

1  para j ← 0, …, k:  c_j ← cuantil empírico de orden j/k de los valores sin NaN
2  eliminar los cortes repetidos             ▹ duplicates="drop"
3  k' ← (número de cortes que quedan) − 1    ▹ k' ≤ k intervalos efectivos
4  para cada fila g:
5      si x_g es NaN o k' = 0 entonces y_g ← NaN
6      si no, si x_g = c_0 entonces y_g ← 0  ▹ el primer intervalo es cerrado
7      si no y_g ← el j tal que c_j < x_g ≤ c_{j+1}
8  devolver y
```

- Si los valores son distintos, cada intervalo tiene $\lfloor n/k \rfloor$ o $\lceil n/k \rceil$ objetos (Proposición 1.22), salvo diferencias de un objeto por redondeo en punto flotante (Observación 1.23).
- Con empates, varios cuantiles pueden coincidir y quedan menos de $k$ intervalos. Por ejemplo, con 6 ceros entre 10 valores y $k = 4$, solo quedan 2 códigos distintos.
- Si la variable es constante, todos los cortes coinciden, no queda ningún intervalo y **toda la columna queda en NaN**.

**Complejidad:** $O(n \log n)$, por el cálculo de los cuantiles y la búsqueda del intervalo de cada valor.

### `kmeans_1d(serie, k)`

Agrupa los valores con k-medias (algoritmo de Lloyd) y corta en los puntos medios entre centroides consecutivos (Definición 1.24, Teorema 1.26 y Proposición 1.28). Es la opción de «k-means unidimensional» que menciona el plan del proyecto.

```text
KMEANS_1D(x, k)
Entrada: serie x (puede tener NaN), entero k ≥ 2
Salida:  código y_g para cada fila g

 1  v ← valores de x sin NaN, en orden ascendente
 2  si |v| ≤ k o mín(v) = máx(v) entonces
 3      devolver 0 para todas las filas de x                  ▹ incluidas las que tienen NaN
 4  para i ← 1, …, k:  μ_i ← percentil(v, 100 · i / (k + 1))  ▹ inicio determinista
 5  repetir a lo sumo 30 veces:
 6      para cada valor v_t:  ℓ_t ← argmin_i |v_t − μ_i|      ▹ paso de asignación
 7      para i ← 1, …, k:                                     ▹ paso de actualización
 8          si algún ℓ_t = i entonces μ'_i ← media{ v_t : ℓ_t = i }
 9          si no μ'_i ← μ_i                                  ▹ un grupo vacío conserva su centroide
10      si μ' ≈ μ entonces salir del ciclo                    ▹ np.allclose
11      μ ← μ'
12  ordenar los centroides: μ_1 ≤ … ≤ μ_k
13  para i ← 1, …, k − 1:  b_i ← (μ_i + μ_{i+1}) / 2          ▹ fronteras entre grupos
14  cortes ← valores distintos de (−∞, b_1, …, b_{k−1}, +∞)
15  para cada fila g:
16      si x_g es NaN entonces y_g ← NaN
17      si no y_g ← el j tal que cortes_j < x_g ≤ cortes_{j+1}
18  devolver y
```

- **Inicialización determinista.** Los centroides iniciales son percentiles equiespaciados (25, 50 y 75 con $k = 3$), así que el resultado no depende de una semilla aleatoria y es reproducible.
- **Convergencia.** El Teorema 1.26 garantiza que la suma de cuadrados no aumenta y que los centroides se estabilizan en un número finito de pasos. En el código, la parada se detecta con `np.allclose` (una tolerancia de punto flotante), y el tope de 30 iteraciones asegura que termina aunque no se alcance esa tolerancia. El resultado es un óptimo local, no necesariamente el global (Observación 1.27).
- **Cortes en los puntos medios.** Por la Proposición 1.28, cortar en $b_1, \dots, b_{k-1}$ equivale a asignar cada valor a su centroide más cercano. Si dos centroides coinciden, `np.unique` elimina el corte repetido y quedan menos de $k$ intervalos.
- **Caso trivial.** Si hay $k$ o menos valores válidos, o todos son iguales, la función devuelve 0 en todas las filas, **incluidas las que tenían NaN**. Es la única rama de los tres métodos que no conserva los NaN.

**Complejidad:** $O(n \log n)$ para ordenar, más $O(T \cdot n k)$ para las $T \le 30$ iteraciones. La matriz de distancias de cada iteración ocupa $n \times k$ flotantes.

## Los tres métodos sobre un mismo ejemplo

Radio de los seis objetos del ejemplo de la §1.6 de los [fundamentos teóricos](../../teoria/teoria.pdf), con $k = 3$:

| Objeto | radio | `equal_width` | `equal_frequency` | `kmeans_1d` |
|:-:|:-:|:-:|:-:|:-:|
| $g_1$ | 12.0 | 0 | 0 | 0 |
| $g_2$ | 13.5 | 0 | 1 | 1 |
| $g_3$ | 20.1 | 2 | 2 | 2 |
| $g_4$ | 11.2 | 0 | 0 | 0 |
| $g_5$ | 17.8 | 2 | 2 | 2 |
| $g_6$ | 14.6 | 1 | 1 | 1 |
| **Cortes interiores** | | 14.17 y 17.13 | 13.00 y 15.67 | 12.83 y 16.50 |

Con igual ancho, $g_2$ queda con los valores pequeños; con igual frecuencia y con k-medias queda en el intervalo central. Igual frecuencia deja exactamente dos objetos por intervalo, como predice la Proposición 1.22 con $n/k = 2$. k-medias converge en dos iteraciones: los centroides iniciales son 12.375, 14.05 y 17.0, y los finales 11.6, 14.05 y 18.95, cuyos puntos medios son los cortes de la tabla.

```python
import pandas as pd
from src.preprocessing.cuant_type import cut_methods

radio = pd.Series([12.0, 13.5, 20.1, 11.2, 17.8, 14.6])
for nombre, cortar in cut_methods.items():
    print(nombre, cortar(radio, 3).tolist())
# equal_width [0, 0, 2, 0, 2, 1]
# equal_frequency [0, 1, 2, 0, 2, 1]
# kmeans_1d [0, 1, 2, 0, 2, 1]
```

## Casos borde

| Situación | `equal_width` | `equal_frequency` | `kmeans_1d` |
|---|---|---|---|
| NaN en la serie | Se conservan como NaN | Se conservan como NaN | Se conservan, salvo en el caso trivial (0) |
| Serie completamente NaN | `ValueError` | Todo NaN | Todo 0 |
| Variable constante | Mismo código para todos | **Todo NaN** | Todo 0 |
| Un solo valor válido | Ese valor recibe un código | Todo NaN | Todo 0 |
| $k$ o menos valores válidos | Normal, con intervalos vacíos | Normal, con intervalos vacíos | Todo 0 |
| Muchos empates | Intervalos vacíos | Menos de $k$ intervalos | Menos de $k$ intervalos si coinciden centroides |
