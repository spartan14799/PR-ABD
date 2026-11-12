"""Dibujo del diagrama de Hasse de un retículo de conceptos.

Recibe los conceptos y las aristas que calcula `make_lattice` (fca/lattice.py)
y dibuja el diagrama con matplotlib, con etiquetado reducido: cada atributo se
escribe una sola vez, sobre su concepto atributo, y cada objeto una sola vez,
debajo de su concepto objeto.

Ejemplo:

    import matplotlib.pyplot as plt
    from fca.lattice import make_lattice
    from fca.lattice_plot import plot_lattice

    concepts, edges = make_lattice(I)          # I: matriz de incidencia (DataFrame)
    plot_lattice(concepts, edges, I.index, I.columns, title="Mi retículo")
    plt.show()                                 # o plt.savefig("reticulo.png")
"""
from collections import defaultdict
import textwrap
import warnings

import matplotlib.pyplot as plt
import numpy as np

# Por encima de este número de conceptos el diagrama deja de ser legible.
MAX_CONCEPTOS_LEGIBLE = 150


def _levels(concepts, edges):
    down = defaultdict(list)
    for i, j in edges:
        down[i].append(j)
    level = [0] * len(concepts)
    for k in sorted(range(len(concepts)), key=lambda k: -concepts[k][0].bit_count()):
        for j in down[k]:
            level[j] = max(level[j], level[k] + 1)
    return level


def _segments(pos, edges):
    P = np.array([pos[i] for i, _ in edges], dtype=float)
    Q = np.array([pos[j] for _, j in edges], dtype=float)
    return P, Q


def _crossings(pos, edges):
    if len(edges) < 2:
        return 0
    P, Q = _segments(pos, edges)

    def orient(A, B, C):
        v = ((B[..., 0] - A[..., 0]) * (C[..., 1] - A[..., 1])
             - (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0]))
        return np.where(np.abs(v) < 1e-9, 0, np.sign(v))

    p1, p2 = P[:, None], Q[:, None]
    q1, q2 = P[None, :], Q[None, :]
    cross = ((orient(p1, p2, q1) * orient(p1, p2, q2) < 0)
             & (orient(q1, q2, p1) * orient(q1, q2, p2) < 0))
    return int(np.triu(cross, 1).sum())


def _edges_through_nodes(pos, edges, tol=0.25):
    if not edges:
        return 0
    P, Q = _segments(pos, edges)
    N = np.array(list(pos.values()), dtype=float)
    x1, y1, x2, y2 = P[:, :1], P[:, 1:], Q[:, :1], Q[:, 1:]
    nx, ny = N[None, :, 0], N[None, :, 1]
    between = (ny > np.minimum(y1, y2)) & (ny < np.maximum(y1, y2))
    xs = x1 + (x2 - x1) * (ny - y1) / (y2 - y1)   # y1 != y2 siempre
    return int((between & (np.abs(xs - nx) < tol)).sum())


def _score(pos, edges, overlap_weight=1):
    return _crossings(pos, edges) + overlap_weight * _edges_through_nodes(pos, edges)


def hasse_layout(concepts, edges, sweeps=16, max_edges_for_check=2500,
                 max_edges_for_swaps=300):
    """Posición (x, y) de cada concepto en el diagrama.

    Cada concepto va en un nivel (y = -nivel) según el camino más largo desde
    el concepto superior, así que toda arista baja. Dentro de cada nivel, el
    orden se elige para que haya pocos cruces y ninguna arista pase por encima
    de un nodo: primero con barridos de baricentros, luego intercambiando
    vecinos (solo si hay a lo sumo `max_edges_for_swaps` aristas) y al final
    desplazando niveles completos. Con más de `max_edges_for_check` aristas no
    se cuentan cruces, porque contarlos cuesta O(aristas²).

    Devuelve un diccionario {índice del concepto: (x, y)}.
    """
    level = _levels(concepts, edges)
    up, down = defaultdict(list), defaultdict(list)
    for i, j in edges:
        down[i].append(j)
        up[j].append(i)

    layers = defaultdict(list)
    for k in range(len(concepts)):
        layers[level[k]].append(k)
    depth = max(layers) if layers else 0

    shift = defaultdict(float)

    def place(layers):
        pos = {}
        for l, nodes in layers.items():
            off = (len(nodes) - 1) / 2 - shift[l]
            for r, k in enumerate(nodes):
                pos[k] = (r - off, -l)
        return pos

    pos = place(layers)
    check = len(edges) <= max_edges_for_check
    best = ({l: list(v) for l, v in layers.items()},
            _score(pos, edges) if check else None)

    for s in range(sweeps):
        downward = s % 2 == 0
        order = range(1, depth + 1) if downward else range(depth - 1, -1, -1)
        nbrs = up if downward else down
        for l in order:
            nodes = layers[l]

            def bary(k):
                ns = nbrs[k]
                return sum(pos[n][0] for n in ns) / len(ns) if ns else pos[k][0]

            layers[l] = sorted(nodes, key=bary)
            pos = place(layers)
        if check:
            c = _score(pos, edges)
            if c < best[1]:
                best = ({l: list(v) for l, v in layers.items()}, c)
        else:
            best = ({l: list(v) for l, v in layers.items()}, None)

    layers, score = best
    if score and len(edges) <= max_edges_for_swaps:
        improved = True
        while improved:
            improved = False
            for l, nodes in layers.items():
                for r in range(len(nodes) - 1):
                    nodes[r], nodes[r + 1] = nodes[r + 1], nodes[r]
                    c = _score(place(layers), edges)
                    if c < score:
                        score, improved = c, True
                    else:
                        nodes[r], nodes[r + 1] = nodes[r + 1], nodes[r]

    if check and _edges_through_nodes(place(layers), edges):
        offsets = (0, -0.5, 0.5, -0.25, 0.25, -0.75, 0.75)
        for _ in range(3):
            for l in layers:
                cands = []
                for o in offsets:
                    shift[l] = o
                    cands.append((_score(place(layers), edges, 50), abs(o), o))
                shift[l] = min(cands)[2]

    return place(layers)



def reduced_labels(concepts, objects, attributes):
    """Etiquetado reducido: dónde se escribe cada atributo y cada objeto.

    El atributo m va en su concepto atributo (m', m''), que es el concepto de
    mayor extensión cuya intensión contiene a m. El objeto g va en su concepto
    objeto (g'', g'), que es el concepto de menor extensión que contiene a g.

    Devuelve dos diccionarios {índice del concepto: [nombres]}: uno para los
    atributos y otro para los objetos.
    """
    att_lab, obj_lab = defaultdict(list), defaultdict(list)
    for m, name in enumerate(attributes):
        cand = [k for k, (_, B) in enumerate(concepts) if (B >> m) & 1]
        if cand:
            k = max(cand, key=lambda k: concepts[k][0].bit_count())
            att_lab[k].append(str(name))
    for g, name in enumerate(objects):
        cand = [k for k, (A, _) in enumerate(concepts) if (A >> g) & 1]
        if cand:
            k = min(cand, key=lambda k: concepts[k][0].bit_count())
            obj_lab[k].append(str(name))
    return att_lab, obj_lab



def plot_lattice(concepts, edges, objects, attributes, ax=None, *,
                 labels="reduced", show_index=False, title=None,
                 node_size=11, font_size=8, wrap=28,
                 att_color="#2a6fdb", obj_color="#1f1f1f",
                 edge_color="#a7acb3", figsize=None):
    """Dibuja el diagrama de Hasse y devuelve el `Axes` de matplotlib.

    Parámetros:
        concepts, edges: el resultado de `make_lattice`.
        objects, attributes: los nombres, en el mismo orden que las filas y
            columnas de la matriz de incidencia (por ejemplo, I.index y I.columns).
        ax: un `Axes` donde dibujar; si es None se crea una figura nueva.
        labels: "reduced" para el etiquetado reducido o "none" para no escribir
            nombres.
        show_index: escribe al lado de cada nodo su índice en `concepts`.

    La mitad superior de un nodo se pinta si ahí se escribe algún atributo, y
    la inferior si ahí se escribe algún objeto. Para verlo hay que llamar a
    plt.show() o guardarlo con plt.savefig(...).
    """
    if labels not in ("reduced", "none"):
        raise ValueError(f"labels debe ser 'reduced' o 'none', no {labels!r}")
    if len(concepts) > MAX_CONCEPTOS_LEGIBLE:
        warnings.warn(
            f"El retículo tiene {len(concepts)} conceptos: el diagrama será difícil "
            "de leer y puede tardar. Conviene dibujar contextos más pequeños.",
            stacklevel=2,
        )
    objects, attributes = list(objects), list(attributes)
    pos = hasse_layout(concepts, edges)

    if ax is None:
        width = max(len([p for p in pos.values() if p[1] == y])
                    for y in {p[1] for p in pos.values()})
        depth = -min(p[1] for p in pos.values())
        figsize = figsize or (max(6, 1.6 * width), max(4, 1.4 * (depth + 1)))
        _, ax = plt.subplots(figsize=figsize)

    for i, j in edges:
        (x1, y1), (x2, y2) = pos[i], pos[j]
        ax.plot([x1, x2], [y1, y2], color=edge_color, lw=1.1, zorder=1,
                solid_capstyle="round")

    att_lab, obj_lab = reduced_labels(concepts, objects, attributes)

    for k, (x, y) in pos.items():
        ax.plot(x, y, marker="o", markersize=node_size, fillstyle="top",
                markerfacecolor=att_color if att_lab[k] else "white",
                markerfacecoloralt=obj_color if obj_lab[k] else "white",
                markeredgecolor="#555b63", markeredgewidth=1, zorder=2)
        if show_index:
            ax.annotate(str(k), (x, y), xytext=(node_size * 0.6, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=font_size - 1, color="#8a9099", zorder=3)

    if labels == "reduced":
        box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
        off = node_size * 0.55 + 2
        for k, names in att_lab.items():
            if names:
                ax.annotate(textwrap.fill(", ".join(names), wrap), pos[k],
                            xytext=(0, off), textcoords="offset points",
                            ha="center", va="bottom", fontsize=font_size,
                            color=att_color, bbox=box, zorder=4)
        for k, names in obj_lab.items():
            if names:
                ax.annotate(textwrap.fill(", ".join(names), wrap), pos[k],
                            xytext=(0, -off), textcoords="offset points",
                            ha="center", va="top", fontsize=font_size,
                            color=obj_color, bbox=box, zorder=4)

    if title:
        ax.set_title(title)
    ax.margins(x=0.12, y=0.12)
    ax.set_axis_off()
    return ax
