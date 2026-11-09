from collections import defaultdict
from fca.context import bits

Implication = tuple[int, int]


def attribute_closure(X: int, F: list[Implication]) -> int:
    need = [A.bit_count() for A, _ in F]
    by_att = defaultdict(list)
    for k, (A, _) in enumerate(F):
        for j in bits(A):
            by_att[j].append(k)

    W = X
    queue = list(bits(X))

    for k, (A, B) in enumerate(F):
        if A == 0:
            new = B & ~W
            W |= new
            queue.extend(bits(new))

    while queue:
        j = queue.pop()
        for k in by_att[j]:
            need[k] -= 1
            if need[k] == 0:
                new = F[k][1] & ~W
                if new:
                    W |= new
                    queue.extend(bits(new))
    return W


class ImplicationClosure:
    def __init__(self) -> None:
        self.F: list[Implication] = []
        self.size: list[int] = []
        self.by_att: dict[int, list[int]] = defaultdict(list)
        self.empty: int = 0            # unión de las conclusiones con premisa vacía

    def add(self, A: int, B: int) -> None:
        k = len(self.F)
        self.F.append((A, B))
        self.size.append(A.bit_count())
        if A == 0:
            self.empty |= B
        for j in bits(A):
            self.by_att[j].append(k)

    def __call__(self, X: int) -> int:
        F, size, by_att = self.F, self.size, self.by_att

        # Contadores solo para las implicaciones que realmente se tocan
        # (antes se copiaba la lista completa `size` en cada llamada).
        need: dict[int, int] = {}
        get = need.get

        W = X | self.empty
        queue = list(bits(W))
        while queue:
            j = queue.pop()
            for k in by_att.get(j, ()):
                c = get(k, size[k]) - 1
                if c:
                    need[k] = c
                else:
                    new = F[k][1] & ~W
                    if new:
                        W |= new
                        queue.extend(bits(new))
        return W
