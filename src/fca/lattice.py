import pandas as pd
from .context import FormalContext, bits


def make_lattice(incidence_matrix):
    fc = FormalContext(incidence_matrix)

    concepts = []
    A = fc.closure(0)
    while A is not None:
        concepts.append((fc.extent(A), A))
        A = fc.next_closure(A)

    index = {ext: i for i, (ext, _) in enumerate(concepts)}

    edges = []
    for i, (A, B) in enumerate(concepts):
        cand = {A & fc.ext[m] for m in bits(fc.full_att & ~B)}
        kept = []
        for c in sorted(cand, key=int.bit_count, reverse=True):
            if not any(c & k == c for k in kept):
                kept.append(c)
        edges += [(i, index[c]) for c in kept]

    return concepts, edges
