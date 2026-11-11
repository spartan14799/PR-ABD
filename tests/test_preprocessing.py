"""Pruebas de discretización y escalado.

Usan el mismo ejemplo de la documentación (docs/codigo/preprocessing/ y §1.6 de
la teoría), así que si el código cambia y el ejemplo deja de ser cierto, falla aquí.
"""
import numpy as np
import pandas as pd

from preprocessing.cuant_type import cut_methods
from preprocessing.quantiles import methods
from preprocessing.scaler import discretize, scaling

RADIO = pd.Series([12.0, 13.5, 20.1, 11.2, 17.8, 14.6])


def tabla_ejemplo() -> pd.DataFrame:
    return pd.DataFrame(
        {"radio": RADIO.tolist(),
         "textura": [15.7, 21.3, 17.8, 25.0, 19.9, 14.1],
         "diagnostico": ["B", "B", "M", "B", "M", "B"]},
        index=["g1", "g2", "g3", "g4", "g5", "g6"],
    )


def test_reglas_para_el_numero_de_intervalos():
    resultado = {nombre: regla(RADIO) for nombre, regla in methods.items()}
    assert resultado == {"sturges": 4, "freedman": 2, "scott": 2}


def test_metodos_de_corte():
    assert cut_methods["equal_width"](RADIO, 3).tolist() == [0, 0, 2, 0, 2, 1]
    assert cut_methods["equal_frequency"](RADIO, 3).tolist() == [0, 1, 2, 0, 2, 1]
    assert cut_methods["kmeans_1d"](RADIO, 3).tolist() == [0, 1, 2, 0, 2, 1]


def test_discretize_solo_cambia_las_columnas_numericas():
    disc, ks = discretize(tabla_ejemplo(), "equal_width", 3)
    assert ks == {"radio": 3, "textura": 3}
    assert disc["radio"].tolist() == [0, 0, 2, 0, 2, 1]
    assert disc["textura"].tolist() == [0, 1, 1, 2, 1, 0]
    assert disc["diagnostico"].tolist() == ["B", "B", "M", "B", "M", "B"]


def test_matriz_de_incidencia_del_ejemplo():
    disc, _ = discretize(tabla_ejemplo(), "equal_width", 3)
    K = pd.concat(
        [scaling(disc[["radio", "textura"]], "ordinal", "le"),
         scaling(disc[["diagnostico"]], "nominal")],
        axis=1,
    )
    assert list(K.columns) == [
        "radio<=0", "radio<=1", "radio<=2",
        "textura<=0", "textura<=1", "textura<=2",
        "diagnostico=B", "diagnostico=M",
    ]
    assert K.astype(int).T.to_dict("list") == {
        "g1": [1, 1, 1, 1, 1, 1, 1, 0],
        "g2": [1, 1, 1, 0, 1, 1, 1, 0],
        "g3": [0, 0, 1, 0, 1, 1, 0, 1],
        "g4": [1, 1, 1, 0, 0, 1, 1, 0],
        "g5": [0, 0, 1, 0, 1, 1, 0, 1],
        "g6": [0, 1, 1, 1, 1, 1, 1, 0],
    }


def test_los_codigos_conservan_el_orden():
    """Proposición 1.19: si x ≤ y, el código de x es menor o igual que el de y."""
    rng = np.random.default_rng(14)
    for _ in range(100):
        x = pd.Series(np.round(rng.exponential(size=int(rng.integers(5, 200))) * 10, 1))
        k = int(rng.integers(2, 10))
        orden = np.argsort(x.to_numpy(), kind="stable")
        for cortar in cut_methods.values():
            codigos = cortar(x, k).to_numpy()
            assert np.all(np.diff(codigos[orden]) >= 0)


def test_escala_ordinal_no_pierde_informacion():
    """Proposición 1.16: el valor es el menor v tal que la fila tiene col<=v."""
    rng = np.random.default_rng(15)
    D = pd.DataFrame({"a": rng.integers(0, 6, 50), "b": rng.integers(0, 3, 50)})
    K = scaling(D, "ordinal", "le")
    for col in D.columns:
        niveles = np.unique(D[col])
        bloque = K[[f"{col}<={v}" for v in niveles]].to_numpy()
        assert np.array_equal(niveles[bloque.argmax(axis=1)], D[col].to_numpy())


def test_escala_nominal_da_un_atributo_por_variable():
    """Proposición 1.12: cada fila tiene exactamente un atributo de cada variable."""
    D = pd.DataFrame({"a": [0, 1, 2, 1], "b": ["x", "y", "x", "x"]})
    K = scaling(D, "nominal")
    for col in D.columns:
        bloque = K[[c for c in K.columns if c.startswith(f"{col}=")]]
        assert (bloque.sum(axis=1) == 1).all()
