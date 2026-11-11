"""Pruebas de la clausura de atributos respecto a un conjunto de implicaciones."""
import random

from implications.armstrong import ImplicationClosure, attribute_closure
from utilidades import clausura_punto_fijo

a, b, c, d = 1, 2, 4, 8   # un bit por atributo


def test_transitividad():
    F = [(a, b), (b, c)]
    assert attribute_closure(a, F) == a | b | c
    assert attribute_closure(d, F) == d


def test_premisa_con_varios_atributos():
    F = [(a | b, c)]
    assert attribute_closure(a, F) == a              # falta b para aplicar la regla
    assert attribute_closure(a | b, F) == a | b | c


def test_premisa_vacia():
    F = [(0, d)]                                     # ∅ → d: d está en toda clausura
    assert attribute_closure(0, F) == d
    ic = ImplicationClosure()
    ic.add(0, d)
    assert ic(a) == a | d


def test_las_dos_implementaciones_coinciden_con_el_punto_fijo():
    rng = random.Random(13)
    for _ in range(500):
        m = rng.randint(1, 8)
        F = [(rng.getrandbits(m) & rng.getrandbits(m), rng.getrandbits(m))
             for _ in range(rng.randint(0, 8))]
        ic = ImplicationClosure()
        for A, B in F:
            ic.add(A, B)
        for _ in range(10):
            X = rng.getrandbits(m)
            assert attribute_closure(X, F) == ic(X) == clausura_punto_fijo(X, F)
