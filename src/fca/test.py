from context import FormalContext, bits
import pandas as pd
from lattice import make_lattice

I = pd.DataFrame(
    {"Bird": [0, 1, 1, 1], "Fish": [1, 0, 0, 1], "Low": [1, 1, 0, 1]},
    dtype=bool,
)
concepts, edges = make_lattice(I)

print(edges)
print(concepts)

assert len(concepts) == 6
assert set(edges) == {(0, 1), (0, 3), (1, 2), (1, 4), (3, 4), (2, 5), (4, 5)}     # fondo: solo el objeto 3, todos los atributos
