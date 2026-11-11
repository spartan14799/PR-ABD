"""Pruebas de la base canónica de Guigues-Duquenne."""
import pandas as pd

from fca.context import FormalContext
from implications.guigues_duquenne import format_implications, guigues_duquenne
from utilidades import (
    clausura_punto_fijo,
    matrices_aleatorias,
    pseudo_intensiones_fuerza_bruta,
)


def test_ejemplo_a_implica_b():
    # Todo objeto con a tiene b, pero no al revés: la base es {a} → {b}
    I = pd.DataFrame({"a": [1, 0, 0], "b": [1, 1, 0]}, dtype=bool)
    fc = FormalContext(I)
    assert format_implications(guigues_duquenne(fc), fc) == ["{a} → {b}"]


def test_atributo_universal_da_premisa_vacia():
    # Si todos los objetos tienen c, la base es ∅ → {c} (Corolario 1.14)
    I = pd.DataFrame({"a": [1, 0], "c": [1, 1]}, dtype=bool)
    fc = FormalContext(I)
    assert format_implications(guigues_duquenne(fc), fc) == ["{∅} → {c}"]


def test_premisas_son_las_pseudo_intensiones():
    for I in matrices_aleatorias(300, semilla=11):
        fc = FormalContext(I)
        base = guigues_duquenne(fc)
        assert sorted(A for A, _ in base) == sorted(pseudo_intensiones_fuerza_bruta(fc))
        for A, B in base:
            assert B == fc.closure(A) & ~A        # la conclusión es A'' sin A


def test_base_completa_y_no_redundante():
    for I in matrices_aleatorias(300, semilla=12):
        fc = FormalContext(I)
        base = guigues_duquenne(fc)
        # Completa: con la base se deduce la clausura de cualquier conjunto
        for X in range(1 << fc.n_att):
            assert clausura_punto_fijo(X, base) == fc.closure(X)
        # No redundante: ninguna implicación se deduce de las demás
        for k, (A, B) in enumerate(base):
            resto = base[:k] + base[k + 1:]
            assert clausura_punto_fijo(A, resto) & B != B
