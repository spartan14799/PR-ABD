from fca.context import FormalContext, bits
from implications.armstrong import attribute_closure, ImplicationClosure, Implication


def guigues_duquenne(fc: FormalContext) -> list:
    ic = ImplicationClosure()
    A = ic(0)
    while A is not None:
        C = fc.closure(A)
        if C != A:
            ic.add(A, C & ~A)
        A = fc.next_closure(A, ic)
    return ic.F


def implication_to_str(impl: Implication, fc: FormalContext,
                       sep: str = ", ", arrow: str = " → ") -> str:
    A, B = impl
    prem = sep.join(str(fc.attributes[i]) for i in bits(A)) or "∅"
    conc = sep.join(str(fc.attributes[i]) for i in bits(B)) or "∅"
    return f"{{{prem}}}{arrow}{{{conc}}}"


def format_implications(L: list[Implication], fc: FormalContext) -> list[str]:
    return [implication_to_str(impl, fc) for impl in L]
