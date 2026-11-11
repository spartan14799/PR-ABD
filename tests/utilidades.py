"""Funciones auxiliares para las pruebas.

Incluye un generador de contextos aleatorios y versiones de fuerza bruta de los
algoritmos, que son lentas pero obviamente correctas y sirven de referencia.
"""
import random

import numpy as np
import pandas as pd


def matrices_aleatorias(cantidad: int, semilla: int, max_obj: int = 9, max_att: int = 7):
    """Genera matrices de incidencia aleatorias, siempre las mismas para una semilla.

    Pueden tener 0 objetos o 0 atributos, para probar también los casos borde.
    """
    rng = random.Random(semilla)
    for _ in range(cantidad):
        n = rng.randint(0, max_obj)
        m = rng.randint(0, max_att)
        p = rng.choice([0.2, 0.5, 0.8])
        datos = np.array(
            [[rng.random() < p for _ in range(m)] for _ in range(n)], dtype=bool
        ).reshape(n, m)
        yield pd.DataFrame(
            datos,
            index=[f"g{i}" for i in range(n)],
            columns=[f"m{j}" for j in range(m)],
        )


def intensiones_fuerza_bruta(fc) -> set[int]:
    """Todas las intensiones: la clausura B'' de cada subconjunto B de atributos."""
    return {fc.closure(B) for B in range(1 << fc.n_att)}


def todas_las_intensiones(fc) -> list[int]:
    """Intensiones en el orden en que las produce NextClosure."""
    resultado, A = [], fc.closure(0)
    while A is not None:
        resultado.append(A)
        A = fc.next_closure(A)
    return resultado


def es_lecticamente_menor(A: int, B: int) -> bool:
    """A < B en orden léctico: el atributo de menor índice en que difieren está en B."""
    diferencia = A ^ B
    if diferencia == 0:
        return False
    menor = diferencia & -diferencia
    return bool(B & menor)


def cobertura_fuerza_bruta(extensiones: list[int]) -> set[tuple[int, int]]:
    """Pares (i, j) tales que la extensión j está justo debajo de la i en el retículo."""
    pares = set()
    for i, a in enumerate(extensiones):
        for j, b in enumerate(extensiones):
            if b == a or b & a != b:
                continue                                   # b no está contenida en a
            entre = any(c not in (a, b) and b & c == b and c & a == c for c in extensiones)
            if not entre:
                pares.add((i, j))
    return pares


def pseudo_intensiones_fuerza_bruta(fc) -> list[int]:
    """Pseudo-intensiones según la definición: P no es cerrado y contiene la
    clausura de toda pseudo-intensión Q contenida estrictamente en P."""
    pseudo = []
    for P in sorted(range(1 << fc.n_att), key=int.bit_count):
        if fc.closure(P) == P:
            continue
        if all(fc.closure(Q) & ~P == 0 for Q in pseudo if Q & P == Q and Q != P):
            pseudo.append(P)
    return pseudo


def clausura_punto_fijo(X: int, implicaciones) -> int:
    """Clausura de X: aplica las implicaciones (A, B) hasta que nada cambie."""
    W = X
    cambio = True
    while cambio:
        cambio = False
        for A, B in implicaciones:
            if A & W == A and B & ~W:
                W |= B
                cambio = True
    return W
