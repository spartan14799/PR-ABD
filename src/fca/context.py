from typing import Hashable, Iterable, Iterator, Sequence, Union, Optional

import numpy as np
import pandas as pd


def bits(x: int) -> Iterator[int]:
    """Índices de los bits en 1 de x, de menor a mayor.

    Usa la representación binaria como cadena y str.find (implementado en C),
    así el coste total es O(tamaño de x) en lugar de O(popcount * tamaño de x)
    como ocurría con el truco x ^= x & -x sobre enteros grandes.
    """
    s = bin(x)[:1:-1]          # LSB primero, sin el prefijo '0b'
    i = s.find("1")
    while i != -1:
        yield i
        i = s.find("1", i + 1)


class FormalContext:
    objects: list
    attributes: list

    n_obj: int
    n_att: int

    ext: list[int]        # ext[j]  = máscara de objetos que tienen el atributo j
    itx: list[int]        # itx[g]  = máscara de atributos del objeto g
    not_ext: list[int]    # not_ext[j] = objetos que NO tienen el atributo j
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

        # Un solo packbits por eje (en vez de uno por columna / fila).
        cols = np.packbits(M.T, axis=1, bitorder="little")
        rows = np.packbits(M,   axis=1, bitorder="little")
        self.ext = [int.from_bytes(r.tobytes(), "little") for r in cols]
        self.itx = [int.from_bytes(r.tobytes(), "little") for r in rows]

        self.full_obj = (1 << self.n_obj) - 1
        self.full_att = (1 << self.n_att) - 1

        # Complementos precalculados: e ⊆ ext[j]  <=>  e & not_ext[j] == 0
        self.not_ext = [self.full_obj ^ e for e in self.ext]

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _to_mask(names: Iterable[Hashable], index: dict[Hashable, int]) -> int:
        m = 0
        for n in names:
            m |= 1 << index[n]
        return m

    @staticmethod
    def _to_names(mask: int, labels: list) -> list:
        return [labels[i] for i in bits(mask)]

    # ------------------------------------------------------- operadores básicos
    def intent_from_extent(self, e: int) -> int:
        """Atributos comunes a todos los objetos de la máscara e (operador ').

        Elige la estrategia más barata:
          - pocos objetos: AND de itx[g] sobre los objetos de e;
          - muchos objetos: un AND por atributo (e ⊆ ext[j]), sin recorrer objetos.
        """
        if e.bit_count() < self.n_att:
            r = self.full_att
            itx = self.itx
            for g in bits(e):
                r &= itx[g]
                if not r:
                    break
            return r

        r = 0
        j = 0
        for ne in self.not_ext:
            if not (e & ne):
                r |= 1 << j
            j += 1
        return r

    def extent(self, B: Union[int, Sequence[Hashable]]) -> Union[int, list]:
        human = not isinstance(B, int)
        if human:
            B = self._to_mask(B, self.att_index)

        e = self.full_obj
        ext = self.ext
        for j in bits(B):
            e &= ext[j]
            if not e:
                break

        return self._to_names(e, self.objects) if human else e

    def intent(self, A: Union[int, Sequence[Hashable]]) -> Union[int, list]:
        human = not isinstance(A, int)
        if human:
            A = self._to_mask(A, self.obj_index)

        i = self.intent_from_extent(A)

        return self._to_names(i, self.attributes) if human else i

    def closure(self, B: Union[int, Sequence[Hashable]]) -> Union[int, list]:
        human = not isinstance(B, int)
        if human:
            B = self._to_mask(B, self.att_index)

        c = self.intent_from_extent(self.extent(B))

        return self._to_names(c, self.attributes) if human else c

    # ------------------------------------------------------------- NextClosure
    def _oplus(self, A: int, i: int, cl=None) -> int:
        cl = cl or self.closure
        return cl((A & ((1 << i) - 1)) | (1 << i))

    def next_closure(self, A: int, cl=None) -> Optional[int]:
        cl = cl or self.closure
        for i in range(self.n_att - 1, -1, -1):
            bit = 1 << i
            if A & bit:
                continue
            lower = bit - 1
            B = cl((A & lower) | bit)
            if (B ^ A) & lower == 0:
                return B
        return None
