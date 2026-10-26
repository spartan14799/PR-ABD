import sys
import math
from fca.context import FormalContext, bits

BITS_PRED = 1


class TrieNode:
    """Nodo para la estructura Trie de reglas exactas FCA."""
    def __init__(self):
        self.children = {}
        self.rule = None


def construir_trie_reglas(reglas: dict) -> TrieNode:
    """Construye un Trie binario a partir del conjunto de reglas exactas."""
    root = TrieNode()
    for w, pred in reglas.items():
        curr = root
        for bit in w:
            if bit not in curr.children:
                curr.children[bit] = TrieNode()
            curr = curr.children[bit]
        curr.rule = pred
    return root


def serializar_trie(root: TrieNode) -> str:
    """Serializa el Trie en una cadena ultracompacta de bits.
    
    Estructura por nodo:
    - Nodo Hoja (Regla FCA): '1' + bit_predicho (2 bits en total)
    - Nodo Interno: '0' + existe_hijo_0 + existe_hijo_1 (3 bits en total)
    """
    bits_out = []

    def dfs(n):
        if n.rule is not None:
            bits_out.append('1')
            bits_out.append(n.rule)
        else:
            bits_out.append('0')
            has_0 = '1' if '0' in n.children else '0'
            has_1 = '1' if '1' in n.children else '0'
            bits_out.append(has_0)
            bits_out.append(has_1)
            if '0' in n.children:
                dfs(n.children['0'])
            if '1' in n.children:
                dfs(n.children['1'])

    dfs(root)
    return "".join(bits_out)


def deserializar_trie(bitstr: str) -> dict:
    """Reconstruye el diccionario de reglas a partir de la cabecera comprimida."""
    pos = 0
    reglas = {}

    def dfs(current_w):
        nonlocal pos
        if pos >= len(bitstr):
            return
        is_rule = bitstr[pos]
        pos += 1
        if is_rule == '1':
            target = bitstr[pos]
            pos += 1
            reglas[current_w] = target
        else:
            has_0 = bitstr[pos]
            has_1 = bitstr[pos + 1]
            pos += 2
            if has_0 == '1':
                dfs(current_w + '0')
            if has_1 == '1':
                dfs(current_w + '1')

    dfs("")
    return reglas


def costo_regla_estimado(w: str, b: int) -> int:
    """Costo individual usado para evaluar rentabilidad durante la búsqueda."""
    return len(w) + BITS_PRED + math.ceil(math.log2(b + 1))


def extraer_reglas_guigues_duquenne(s: str, b: int) -> dict:
    """Extrae la base canónica de implicaciones exactas FCA mediante operaciones de extensión bitwise."""
    n = len(s)
    N = n - b - BITS_PRED + 1

    # Identificar patrones de contexto observados
    cuentas = {}
    for i in range(0, N):
        prev = s[i : i + BITS_PRED]
        for m in range(1, b + 1):
            w = s[i + BITS_PRED : i + BITS_PRED + m]
            c = cuentas.setdefault(w, {})
            c[prev] = c.get(prev, 0) + 1

    deterministicas = set()
    reglas = {}

    for w in sorted(cuentas, key=lambda x: (len(x), x)):
        c = cuentas[w]
        # Implicación exacta: solo una consecuencia observada (100% confianza)
        if len(c) > 1:
            continue

        # Poda por Cierre FCA: Si un prefijo ya es regla exacta, este supercontexto es redundante
        if any(w[:k] in deterministicas for k in range(1, len(w))):
            continue

        deterministicas.add(w)
        prediccion = list(c.keys())[0]
        frecuencia = c[prediccion]

        # Condición de rentabilidad
        if frecuencia > costo_regla_estimado(w, b):
            reglas[w] = prediccion

    return dict(sorted(reglas.items(), key=lambda kv: (len(kv[0]), kv[0])))


def bit_deducido(contexto: str, reglas: dict, b: int):
    for m in range(1, b + 1):
        v = reglas.get(contexto[:m])
        if v is not None:
            return v
    return None


def codificar(s: str, b: int, reglas: dict):
    n = len(s)
    datos = []
    i = n - b - BITS_PRED
    while i >= 0:
        v = bit_deducido(s[i + BITS_PRED : i + BITS_PRED + b], reglas, b)
        if v is None:
            datos.append(s[i : i + BITS_PRED])
        i -= BITS_PRED

    if i < -1:
        datos.append(s[0 : i + BITS_PRED])

    return s[n - b :], "".join(datos)


def evaluar(s: str, b: int):
    reglas = extraer_reglas_guigues_duquenne(s, b)
    semilla, datos = codificar(s, b, reglas)

    # Empaquetamiento de la base de reglas FCA mediante Trie Binario
    trie_root = construir_trie_reglas(reglas)
    trie_bits = serializar_trie(trie_root)
    costo_reglas = len(trie_bits)

    total_bits = len(semilla) + len(datos) + costo_reglas
    return reglas, semilla, datos, costo_reglas, total_bits, trie_bits


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("Uso: python test.py BITS [BITS_CONTEXTO]")
    s = "".join(c for c in sys.argv[1] if c in "01")
    n = len(s)
    if n < 2 + BITS_PRED:
        sys.exit(f"Hacen falta al menos {2 + BITS_PRED} bits.")

    if len(sys.argv) == 3:
        opciones = [int(sys.argv[2])]
    else:
        opciones = list(range(1, min(25, n - 1) + 1))

    mejor = None
    if len(opciones) > 1:
        print(" b | reglas | datos | semilla | costo reglas (Trie) | total")

    for b in opciones:
        reglas, semilla, datos, costo, total, trie_bits = evaluar(s, b)
        if len(opciones) > 1:
            print(
                f"{b:2d} | {len(reglas):6d} | {len(datos):5d} | {len(semilla):7d} | "
                f"{costo:19d} | {total}"
            )
        if mejor is None or total < mejor[1][4]:
            mejor = (b, (reglas, semilla, datos, costo, total, trie_bits))

    b, (reglas, semilla, datos, costo, total, trie_bits) = mejor
    print(f"\nContexto elegido: b = {b}")
    print(f"=== REGLAS FCA (base de Guigues-Duquenne empaquetada en Trie) ===")
    if not reglas:
        print("(ninguna regla compensa su costo)")
    for w, v in reglas.items():
        print(f"siguientes {w}  →  anteriores {v}")

    # Verificación de reconstrucción exacta mediante deserialización del Trie
    reglas_recuperadas = deserializar_trie(trie_bits)
    ok_trie = (reglas_recuperadas == reglas)

    print("\n=== RESUMEN ===")
    print(
        f"Original: {n} bits | semilla: {len(semilla)} | datos: {len(datos)} | "
        f"reglas (Trie): {costo} bits ({len(reglas)} reglas) | total: {total} bits"
    )
    print(f"Verificación de deserialización Trie: {'OK' if ok_trie else 'FALLÓ'}")


if __name__ == "__main__":
    main()
