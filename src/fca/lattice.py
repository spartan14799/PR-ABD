import pandas as pd
from .context import FormalContext, bits


def _lectic_key(b: int, n: int) -> int:
    """Clave de orden léctico para una máscara de atributos b (n atributos).

    A <_léctico B  <=>  el menor atributo en que difieren pertenece a B
                   <=>  el entero con el bit 0 como más significativo es menor.
    """
    return int(bin(b)[:1:-1].ljust(n, "0"), 2)


def concepts_cbo(fc: FormalContext) -> list[tuple[int, int]]:
    """Todos los conceptos (extent, intent) con Close-by-One.

    Arrastra la extensión hacia abajo (un solo AND por candidato) y descarta los
    candidatos no canónicos sin calcular su clausura completa. Al final se
    ordena en orden léctico de intents, que es el mismo orden que produce
    NextClosure, así los índices de los conceptos no cambian.
    """
    n = fc.n_att
    ext, ne = fc.ext, fc.not_ext
    intent_of = fc.intent_from_extent

    top = fc.full_obj
    b_top = intent_of(top)
    out = [(top, b_top)]
    stack = [(top, b_top, 0)]

    while stack:
        e, b, y = stack.pop()
        for j in range(y, n):
            if (b >> j) & 1:
                continue
            e2 = e & ext[j]

            # Test de canonicidad: ningún k < j fuera de b puede entrar
            # en la clausura de e2 (k entra  <=>  e2 ⊆ ext[k]).
            todo = ((1 << j) - 1) & ~b
            while todo:
                low = todo & -todo
                if not (e2 & ne[low.bit_length() - 1]):
                    break                      # no canónico
                todo ^= low
            else:
                b2 = intent_of(e2)
                out.append((e2, b2))
                stack.append((e2, b2, j + 1))

    out.sort(key=lambda c: _lectic_key(c[1], n))
    return out


def make_lattice(incidence_matrix):
    fc = FormalContext(incidence_matrix)

    concepts = concepts_cbo(fc)
    index = {e: i for i, (e, _) in enumerate(concepts)}

    ext = fc.ext
    full_att = fc.full_att

    edges = []
    for i, (A, B) in enumerate(concepts):
        cand = {A & ext[m] for m in bits(full_att & ~B)}
        kept = []
        for c in sorted(cand, key=int.bit_count, reverse=True):
            for k in kept:
                if c & k == c:
                    break
            else:
                kept.append(c)
        edges += [(i, index[c]) for c in kept]

    return concepts, edges
