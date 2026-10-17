from typing import Hashable, Iterable, Iterator, Sequence, Union

import numpy as np
import pandas as pd


def bits(x: int) -> Iterator[int]:
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low

class FormalContext:
    objects: list
    attributes: list

    n_obj: int
    n_att: int

    ext: list[int]
    itx: list[int]
    full_obj: int
    full_att: int

    att_index: dict[Hashable, int]
    obj_index: dict[Hashable, int]

    def __init__(self, incidence_matrix: pd.DataFrame) -> None:
        self.attributes = list(incidence_matrix.columns)
        self.objects    = list(incidence_matrix.index)

        self.n_obj = len(self.objects)
        self.n_att = len(self.attributes)

        self.att_index = {a: j for j, a in enumerate(self.attributes)}
        self.obj_index = {o: g for g, o in enumerate(self.objects)}

        M = incidence_matrix.to_numpy(dtype=bool)
        self.ext = [
            int.from_bytes(np.packbits(M[:, j], bitorder="little").tobytes(), "little")
            for j in range(self.n_att)
        ]
        self.itx = [
            int.from_bytes(np.packbits(M[g, :], bitorder="little").tobytes(), "little")
            for g in range(self.n_obj)
        ]

        self.full_obj = (1 << self.n_obj) - 1
        self.full_att = (1 << self.n_att) - 1

    @staticmethod
    def _to_mask(names: Iterable[Hashable], index: dict[Hashable, int]) -> int:
        m = 0
        for n in names:
            m |= 1 << index[n]
        return m

    @staticmethod
    def _to_names(mask: int, labels: list) -> list:
        return [labels[i] for i in bits(mask)]

    def extent(self, B: Union[int, Sequence[Hashable]]) -> Union[int, list]:
        human = not isinstance(B, int)
        if human:
            B = self._to_mask(B, self.att_index)

        e = self.full_obj
        for j in bits(B):
            e &= self.ext[j]

        return self._to_names(e, self.objects) if human else e

    def intent(self, A: Union[int, Sequence[Hashable]]) -> Union[int, list]:

        human = not isinstance(A, int)
        if human:
            A = self._to_mask(A, self.obj_index)

        i = self.full_att
        for g in bits(A):
            i &= self.itx[g]

        return self._to_names(i, self.attributes) if human else i

    def closure(self, B: Union[int, Sequence[Hashable]]) -> Union[int, list]:
        human = not isinstance(B, int)
        if human:
            B = self._to_mask(B, self.att_index)

        c = self.intent(self.extent(B))

        return self._to_names(c, self.attributes) if human else c

    def _oplus(self, A: int, i: int) -> int:
        lower = (1 << i) - 1
        return self.closure((A & lower) | (1 << i))

    def next_closure(self, A: int) -> Optional[int]:
        for i in range(self.n_att - 1, -1, -1):
            if A >> i & 1:
                continue
            B = self._oplus(A, i)
            lower = (1 << i) - 1
            if B & lower == A & lower:
                return B
        return None
