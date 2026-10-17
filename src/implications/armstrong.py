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
        self.empty: list[int] = []

    def add(self, A: int, B: int) -> None:
        k = len(self.F)
        self.F.append((A, B))
        self.size.append(A.bit_count())
        if A == 0:
            self.empty.append(B)
        for j in bits(A):
            self.by_att[j].append(k)

    def __call__(self, X: int) -> int:
        need = self.size.copy()
        W = X
        for B in self.empty:
            W |= B
        queue = list(bits(W))
        while queue:
            j = queue.pop()
            for k in self.by_att.get(j, ()):
                need[k] -= 1
                if need[k] == 0:
                    new = self.F[k][1] & ~W
                    if new:
                        W |= new
                        queue.extend(bits(new))
        return W
