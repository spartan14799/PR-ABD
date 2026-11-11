"""Pruebas de FormalContext: operadores de derivación y clausura."""
import random

import pandas as pd

from fca.context import FormalContext, bits
from utilidades import matrices_aleatorias


def contexto_ejemplo() -> FormalContext:
    I = pd.DataFrame(
        {"Bird": [0, 1, 1, 1], "Fish": [1, 0, 0, 1], "Low": [1, 1, 0, 1]},
        index=["o0", "o1", "o2", "o3"],
        dtype=bool,
    )
    return FormalContext(I)


def test_bits_devuelve_los_indices_en_orden():
    assert list(bits(0)) == []
    assert list(bits(0b101101)) == [0, 2, 3, 5]


def test_derivacion_con_nombres():
    fc = contexto_ejemplo()
    assert fc.extent(["Bird"]) == ["o1", "o2", "o3"]
    assert fc.extent(["Bird", "Fish"]) == ["o3"]
    assert fc.intent(["o0"]) == ["Fish", "Low"]
    assert fc.intent(["o1", "o2"]) == ["Bird"]
    assert fc.closure(["Fish"]) == ["Fish", "Low"]   # todo objeto con Fish tiene Low


def test_intensiones_con_las_dos_estrategias():
    """intent_from_extent cambia de estrategia según el tamaño de la extensión;
    las dos deben dar lo mismo que la definición."""
    rng = random.Random(0)
    for I in matrices_aleatorias(200, semilla=1):
        fc = FormalContext(I)
        for _ in range(20):
            e = rng.getrandbits(fc.n_obj) if fc.n_obj else 0
            esperado = fc.full_att
            for g in bits(e):
                esperado &= fc.itx[g]
            assert fc.intent_from_extent(e) == esperado


def test_conexion_de_galois():
    """A ⊆ B'  si y solo si  B ⊆ A'."""
    rng = random.Random(2)
    for I in matrices_aleatorias(200, semilla=3):
        fc = FormalContext(I)
        for _ in range(20):
            A = rng.getrandbits(fc.n_obj) if fc.n_obj else 0
            B = rng.getrandbits(fc.n_att) if fc.n_att else 0
            assert (A & fc.extent(B) == A) == (B & fc.intent(A) == B)


def test_propiedades_de_la_clausura():
    """La clausura es extensiva, idempotente y monótona."""
    rng = random.Random(4)
    for I in matrices_aleatorias(200, semilla=5):
        fc = FormalContext(I)
        for B in range(1 << fc.n_att):
            C = fc.closure(B)
            assert B & C == B                 # extensiva: B ⊆ B''
            assert fc.closure(C) == C         # idempotente
        for _ in range(20):
            B2 = rng.getrandbits(fc.n_att) if fc.n_att else 0
            B1 = B2 & (rng.getrandbits(fc.n_att) if fc.n_att else 0)   # B1 ⊆ B2
            assert fc.closure(B1) & fc.closure(B2) == fc.closure(B1)   # monótona
