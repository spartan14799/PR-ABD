"""Pruebas de NextClosure: debe recorrer todas las intensiones, una sola vez
y en orden léctico."""
import pandas as pd

from fca.context import FormalContext
from utilidades import (
    es_lecticamente_menor,
    intensiones_fuerza_bruta,
    matrices_aleatorias,
    todas_las_intensiones,
)


def test_ejemplo_pequeno():
    I = pd.DataFrame(
        {"Bird": [0, 1, 1, 1], "Fish": [1, 0, 0, 1], "Low": [1, 1, 0, 1]},
        dtype=bool,
    )
    fc = FormalContext(I)
    nombres = [fc._to_names(B, fc.attributes) for B in todas_las_intensiones(fc)]
    assert nombres == [[], ["Low"], ["Fish", "Low"], ["Bird"], ["Bird", "Low"], ["Bird", "Fish", "Low"]]


def test_recorre_todas_las_intensiones_una_vez():
    for I in matrices_aleatorias(300, semilla=6):
        fc = FormalContext(I)
        intensiones = todas_las_intensiones(fc)
        assert len(intensiones) == len(set(intensiones))           # sin repeticiones
        assert set(intensiones) == intensiones_fuerza_bruta(fc)    # ninguna falta


def test_respeta_el_orden_lectico():
    for I in matrices_aleatorias(300, semilla=7):
        fc = FormalContext(I)
        intensiones = todas_las_intensiones(fc)
        assert all(es_lecticamente_menor(a, b) for a, b in zip(intensiones, intensiones[1:]))
