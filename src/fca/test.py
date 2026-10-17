from context import FormalContext, bits
import pandas as pd

I = pd.DataFrame(
    {"Bird": [0, 1, 1, 1], "Fish": [1, 0, 0, 1], "Low": [1, 1, 0, 1]},
    dtype=bool,
)
fc = FormalContext(I)

A, out = fc.closure(0), []
while A is not None:
    out.append(A)
    A = fc.next_closure(A)

assert out == [0b000, 0b100, 0b110, 0b001, 0b101, 0b111]
