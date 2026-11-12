# Pruebas

Esta carpeta tiene las pruebas automáticas del proyecto. Comprueban que los algoritmos den resultados correctos y que los sigan dando cuando alguien cambie el código. Se ejecutan con [pytest](https://docs.pytest.org/).

## Cómo ejecutarlas

Desde la raíz del repositorio:

```powershell
python -m pip install -r requirements.txt   # solo la primera vez
python -m pytest                            # todas las pruebas
python -m pytest -v                         # con el nombre de cada prueba
python -m pytest tests/test_lattice.py      # un solo archivo
python -m pytest -k galois                  # las pruebas cuyo nombre contiene «galois»
```

El archivo `pyproject.toml` le indica a pytest que las pruebas están en `tests/` y que agregue `src/` a la ruta de Python. Por eso las pruebas importan con `from fca.context import FormalContext`, sin `src.` adelante.

## Cómo leer el resultado

pytest muestra un carácter por cada prueba:

| Carácter | Significado |
|:-:|---|
| `.` | La prueba pasó: el código dio exactamente lo esperado. |
| `F` | La prueba falló: algún `assert` no se cumplió. |
| `E` | Error: la prueba ni siquiera se pudo ejecutar, por ejemplo por un `ImportError`. |

Al final aparece un resumen como `32 passed in 1.69s`. Si todas pasan, no hay nada que hacer.

Si una falla, pytest muestra qué prueba fue, en qué línea y con qué valores. Por ejemplo, si alguien cambiara `equal_width` para que hiciera un intervalo de más, se vería así:

```
    def test_metodos_de_corte():
>       assert cut_methods["equal_width"](RADIO, 3).tolist() == [0, 0, 2, 0, 2, 1]
E       assert [0, 1, 3, 0, 2, 1] == [0, 0, 2, 0, 2, 1]
E         At index 1 diff: 1 != 0

tests/test_preprocessing.py:31: AssertionError
```

La línea con `>` es el `assert` que no se cumplió; las líneas con `E` muestran lo que dio el código (a la izquierda) y lo que se esperaba (a la derecha).

Los avisos (*warnings*) que aparecen al final no son fallas: informan algo que conviene revisar, pero las pruebas pasaron igual. Por ejemplo, con pytest 9.2 y versiones de la librería anyio anteriores a la 4.15 aparece un aviso sobre `CallSpec2`. Viene de anyio, no del proyecto, y se quita actualizándola con `python -m pip install -U anyio`.

## Qué significa que una prueba pase o falle

- **Si pasa:** en los casos que revisa, el código hace lo que debe. Eso no demuestra que no haya errores en otros casos, pero los casos revisados son muchos (ver la sección siguiente).
- **Si falla:** el código ya no hace lo que hacía o lo que dice la teoría. Hay dos posibilidades:
  1. Se introdujo un error: hay que corregir el código, no la prueba.
  2. El cambio era intencional, por ejemplo si se cambió el nombre de las columnas que genera `scaling`: hay que actualizar la prueba y, casi siempre, la documentación, porque los ejemplos de las pruebas son los mismos de `docs/codigo/`.

  Nunca se borra ni se desactiva una prueba solo para que pase.
- **Falsas alarmas y errores no detectados.** Una prueba puede fallar sin que el código esté mal (un *falso positivo*), por ejemplo si una librería cambia su formato de salida. También puede pasar aunque haya un error (un *falso negativo*), si ninguno de sus casos lo toca. Para reducir este segundo riesgo, estas pruebas se verificaron metiendo errores a propósito en el código: detectaron todos los que se introdujeron en NextClosure, las aristas del retículo, Guigues-Duquenne, la clausura, la discretización, el escalado y el dibujo del diagrama.

## Dos tipos de prueba

1. **Ejemplos pequeños con la respuesta calculada a mano.** Son fáciles de leer y sirven también como ejemplos de uso.
2. **Comparación contra fuerza bruta.** Se generan cientos de contextos aleatorios, de hasta 9 objetos y 7 atributos e incluidos algunos vacíos. En cada uno se compara el algoritmo rápido con una versión lenta pero obviamente correcta, que aplica la definición al pie de la letra. Los contextos salen siempre iguales porque usan una semilla fija, así que si una prueba falla, vuelve a fallar igual y se puede investigar.

Las versiones de fuerza bruta están en `utilidades.py`:

| Función | Qué calcula |
|---|---|
| `matrices_aleatorias` | Matrices de incidencia aleatorias, siempre las mismas para una semilla. |
| `intensiones_fuerza_bruta` | Todas las intensiones: la clausura B'' de cada subconjunto B de atributos. |
| `todas_las_intensiones` | Las intensiones en el orden en que las produce NextClosure. |
| `es_lecticamente_menor` | Si un conjunto va antes que otro en orden léctico. |
| `cobertura_fuerza_bruta` | Las aristas del diagrama de Hasse según la definición: conceptos uno justo debajo del otro. |
| `pseudo_intensiones_fuerza_bruta` | Las pseudo-intensiones según la definición. |
| `clausura_punto_fijo` | La clausura de un conjunto aplicando las implicaciones hasta que nada cambie. |

## Qué comprueba cada archivo

### `test_context.py`: el contexto formal (`fca/context.py`)

| Prueba | Qué comprueba |
|---|---|
| `test_bits_devuelve_los_indices_en_orden` | `bits` devuelve qué bits están en 1; así representa el código los conjuntos. |
| `test_derivacion_con_nombres` | Derivación y clausura con nombres en un ejemplo pequeño: la clausura de {Fish} es {Fish, Low}, porque todo objeto con Fish tiene Low. |
| `test_intensiones_con_las_dos_estrategias` | `intent_from_extent` usa dos estrategias según cuántos objetos haya, y las dos dan lo mismo que la definición. |
| `test_conexion_de_galois` | A ⊆ B′ si y solo si B ⊆ A′. |
| `test_propiedades_de_la_clausura` | La clausura es extensiva, idempotente y monótona. |

### `test_next_closure.py`: NextClosure (`fca/context.py`)

| Prueba | Qué comprueba |
|---|---|
| `test_ejemplo_pequeno` | Las 6 intensiones del ejemplo, en el orden esperado. |
| `test_recorre_todas_las_intensiones_una_vez` | No repite ni se salta ninguna intensión. |
| `test_respeta_el_orden_lectico` | Cada intensión va antes que la siguiente en orden léctico. |

### `test_lattice.py`: el retículo (`fca/lattice.py`)

| Prueba | Qué comprueba |
|---|---|
| `test_ejemplo_pequeno` | 6 conceptos y las 7 aristas esperadas (era la prueba de `src/fca/test.py`). |
| `test_conceptos_coinciden_con_fuerza_bruta` | Close-by-One encuentra todos los conceptos y cada par (extensión, intensión) es de verdad un concepto. |
| `test_mismo_orden_que_next_closure` | Close-by-One da los conceptos en el mismo orden que NextClosure, así que los índices coinciden. |
| `test_aristas_son_la_relacion_de_cobertura` | Cada arista une un concepto con otro justo debajo, sin ninguno en medio, y no falta ninguna. |

### `test_lattice_plot.py`: el dibujo del diagrama de Hasse (`fca/lattice_plot.py`)

| Prueba | Qué comprueba |
|---|---|
| `test_posiciones_validas` | Todo concepto tiene posición, no hay dos nodos en el mismo punto y toda arista baja. |
| `test_etiquetas_reducidas` | Cada atributo se escribe una sola vez, en su concepto atributo (m′, m″), y cada objeto una sola vez, en su concepto objeto (g″, g′). |
| `test_dibuja_el_ejemplo_clasico` | El ejemplo de Ganter y Wille (seres vivos y agua) tiene 19 conceptos y se dibuja con sus nombres. |
| `test_sin_etiquetas` | Con `labels="none"` no se escribe ningún nombre. |
| `test_valor_de_labels_invalido` | Un valor de `labels` que no existe da un error claro. |

### `test_guigues_duquenne.py`: la base canónica (`implications/guigues_duquenne.py`)

| Prueba | Qué comprueba |
|---|---|
| `test_ejemplo_a_implica_b` | Si todo objeto con a tiene b, la base es {a} → {b}. |
| `test_atributo_universal_da_premisa_vacia` | Si todos los objetos tienen c, la base es ∅ → {c} (Corolario 1.14 de la teoría). |
| `test_premisas_son_las_pseudo_intensiones` | Las premisas son exactamente las pseudo-intensiones, y cada conclusión es A″ sin A. |
| `test_base_completa_y_no_redundante` | Con la base se deduce la clausura de cualquier conjunto, y ninguna implicación sale de las demás. |

### `test_armstrong.py`: la clausura de atributos (`implications/armstrong.py`)

| Prueba | Qué comprueba |
|---|---|
| `test_transitividad` | Con a → b y b → c, la clausura de {a} es {a, b, c}. |
| `test_premisa_con_varios_atributos` | Una regla ab → c solo se aplica cuando están a y b. |
| `test_premisa_vacia` | Una regla ∅ → d agrega d siempre. |
| `test_las_dos_implementaciones_coinciden_con_el_punto_fijo` | `attribute_closure` e `ImplicationClosure` dan lo mismo que la versión lenta en 500 conjuntos de implicaciones aleatorios. |

### `test_preprocessing.py`: discretización y escalado (`preprocessing/`)

| Prueba | Qué comprueba |
|---|---|
| `test_reglas_para_el_numero_de_intervalos` | Sturges, Freedman-Diaconis y Scott sobre el ejemplo de la documentación. |
| `test_metodos_de_corte` | Los códigos de igual ancho, igual frecuencia y k-medias en el mismo ejemplo. |
| `test_discretize_solo_cambia_las_columnas_numericas` | `discretize` cambia las columnas numéricas y deja igual las demás. |
| `test_matriz_de_incidencia_del_ejemplo` | La matriz de incidencia completa del ejemplo de la §1.6 de la teoría. |
| `test_los_codigos_conservan_el_orden` | Si x ≤ y, el código de x es menor o igual que el de y (Proposición 1.19). |
| `test_escala_ordinal_no_pierde_informacion` | El valor original se recupera de la fila de la escala ordinal (Proposición 1.16). |
| `test_escala_nominal_da_un_atributo_por_variable` | Cada fila tiene exactamente un atributo de cada variable (Proposición 1.12). |

## Cómo agregar pruebas

- Cada tema va en un archivo `tests/test_<tema>.py`, y cada prueba es una función `test_<lo_que_comprueba>()` que usa `assert`.
- Los módulos se importan desde `src/` sin prefijo: `from fca.lattice import make_lattice`.
- Para un algoritmo nuevo conviene escribir las dos clases de prueba: un ejemplo pequeño hecho a mano y una comparación con la definición en contextos aleatorios, usando `matrices_aleatorias`.
- Antes de cada commit, ejecuten `python -m pytest`.
