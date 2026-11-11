"""Pruebas del retículo: conceptos con Close-by-One y aristas del diagrama de Hasse."""
import pandas as pd

from fca.context import FormalContext
from fca.lattice import concepts_cbo, make_lattice
from utilidades import (
    cobertura_fuerza_bruta,
    intensiones_fuerza_bruta,
    matrices_aleatorias,
    todas_las_intensiones,
)


def test_ejemplo_pequeno():
    # Prueba que antes estaba en src/fca/test.py
    I = pd.DataFrame(
        {"Bird": [0, 1, 1, 1], "Fish": [1, 0, 0, 1], "Low": [1, 1, 0, 1]},
        dtype=bool,
    )
    concepts, edges = make_lattice(I)
    assert len(concepts) == 6
    # El concepto 0 tiene todos los objetos; el 5 solo el objeto 3 y todos los atributos
    assert set(edges) == {(0, 1), (0, 3), (1, 2), (1, 4), (3, 4), (2, 5), (4, 5)}


def test_conceptos_coinciden_con_fuerza_bruta():
    for I in matrices_aleatorias(300, semilla=8):
        fc = FormalContext(I)
        conceptos = concepts_cbo(fc)
        esperadas = intensiones_fuerza_bruta(fc)
        assert len(conceptos) == len(esperadas)
        assert {intension for _, intension in conceptos} == esperadas
        for extension, intension in conceptos:
            assert fc.intent_from_extent(extension) == intension
            assert fc.extent(intension) == extension


def test_mismo_orden_que_next_closure():
    for I in matrices_aleatorias(300, semilla=9):
        fc = FormalContext(I)
        assert [intension for _, intension in concepts_cbo(fc)] == todas_las_intensiones(fc)


def test_aristas_son_la_relacion_de_cobertura():
    for I in matrices_aleatorias(200, semilla=10):
        conceptos, aristas = make_lattice(I)
        extensiones = [extension for extension, _ in conceptos]
        assert set(aristas) == cobertura_fuerza_bruta(extensiones)
