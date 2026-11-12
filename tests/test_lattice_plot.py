"""Pruebas del dibujo del diagrama de Hasse (fca/lattice_plot.py)."""
import matplotlib

matplotlib.use("Agg")            # dibuja sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from fca.context import FormalContext
from fca.lattice import make_lattice
from fca.lattice_plot import hasse_layout, plot_lattice, reduced_labels
from utilidades import matrices_aleatorias


def seres_vivos_y_agua() -> pd.DataFrame:
    """Ejemplo clásico de Ganter y Wille: 8 seres vivos, 9 atributos, 19 conceptos."""
    objetos = ["Leech", "Bream", "Frog", "Dog", "Spike-weed", "Reed", "Bean", "Maize"]
    atributos = ["needs water", "lives in water", "lives on land", "needs chlorophyll",
                 "two seed leaves", "one seed leaf", "can move", "has limbs",
                 "suckles offspring"]
    filas = {"Leech": "abg", "Bream": "abgh", "Frog": "abcgh", "Dog": "acghi",
             "Spike-weed": "abdf", "Reed": "abcdf", "Bean": "acde", "Maize": "acdf"}
    datos = [[letra in filas[o] for letra in "abcdefghi"] for o in objetos]
    return pd.DataFrame(datos, index=objetos, columns=atributos)


def test_posiciones_validas():
    for I in matrices_aleatorias(100, semilla=16):
        conceptos, aristas = make_lattice(I)
        pos = hasse_layout(conceptos, aristas)
        assert set(pos) == set(range(len(conceptos)))           # todos tienen posición
        assert len(set(pos.values())) == len(pos)               # ningún nodo encima de otro
        assert all(pos[i][1] > pos[j][1] for i, j in aristas)   # toda arista baja


def test_etiquetas_reducidas():
    """Cada atributo se escribe una vez, en (m', m''), y cada objeto una vez, en (g'', g')."""
    for I in matrices_aleatorias(100, semilla=17):
        fc = FormalContext(I)
        conceptos, _ = make_lattice(I)
        att_lab, obj_lab = reduced_labels(conceptos, I.index, I.columns)
        for m, nombre in enumerate(I.columns):
            donde = [k for k, nombres in att_lab.items() if nombre in nombres]
            assert len(donde) == 1
            assert conceptos[donde[0]][0] == fc.extent(1 << m)
        for g, nombre in enumerate(I.index):
            donde = [k for k, nombres in obj_lab.items() if nombre in nombres]
            assert len(donde) == 1
            assert conceptos[donde[0]][1] == fc.intent(1 << g)


def test_dibuja_el_ejemplo_clasico():
    I = seres_vivos_y_agua()
    conceptos, aristas = make_lattice(I)
    assert len(conceptos) == 19
    ax = plot_lattice(conceptos, aristas, I.index, I.columns, title="Seres vivos y agua")
    textos = {t.get_text() for t in ax.texts}
    assert {"needs water", "suckles offspring", "Leech", "Maize"} <= textos
    plt.close("all")


def test_sin_etiquetas():
    I = seres_vivos_y_agua()
    conceptos, aristas = make_lattice(I)
    ax = plot_lattice(conceptos, aristas, I.index, I.columns, labels="none")
    assert len(ax.texts) == 0
    plt.close("all")


def test_valor_de_labels_invalido():
    I = seres_vivos_y_agua()
    conceptos, aristas = make_lattice(I)
    with pytest.raises(ValueError):
        plot_lattice(conceptos, aristas, I.index, I.columns, labels="todas")
    plt.close("all")
