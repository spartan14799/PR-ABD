import sys
import os
import math
import time
from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from fca.context import FormalContext

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


# ---------------------------------------------------------------------------
# CONTEXTO FORMAL (FCA) DE LA CADENA
#
#   Objetos   : las posiciones i = 0 .. n-b-1. Cada una tiene su contexto de
#               b bits a la derecha (s[i+1 : i+1+b]) y el bit a predecir s[i].
#   Atributos : "c{j}=v"  -> el j-ésimo bit del contexto vale v   (j<b, v∈{0,1})
#               "y=v"     -> el bit a predecir vale v
#
#   Índices: c{j}=v -> 2*j + v      y=v -> 2*b + v
#
# Una regla exacta "el contexto empieza por w  ->  y = x" es la implicación FCA
#       {c0=w0, ..., c(m-1)=w(m-1)}  ->  {y=x}
# y se cumple  <=>  extent(premisa) ⊆ ext[y=x]  <=>  (y=x) ∈ closure(premisa).
# ---------------------------------------------------------------------------
def construir_contexto_fca(s: str, b: int) -> FormalContext:
    assert BITS_PRED == 1, "el contexto FCA asume BITS_PRED = 1"
    n = len(s)
    N = n - b                                           # n - b - BITS_PRED + 1
    arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8) == ord("1")
    W = sliding_window_view(arr, b + 1)                 # (N, b+1): W[i] = s[i : i+b+1]

    M = np.empty((N, 2 * b + 2), dtype=bool)
    ctx = W[:, 1:]                                      # bits del contexto
    M[:, 0:2 * b:2] = ~ctx                              # c{j}=0
    M[:, 1:2 * b:2] = ctx                               # c{j}=1
    M[:, 2 * b] = ~W[:, 0]                              # y=0
    M[:, 2 * b + 1] = W[:, 0]                           # y=1

    cols = [f"c{j}={v}" for j in range(b) for v in (0, 1)] + ["y=0", "y=1"]
    return FormalContext(pd.DataFrame(M, columns=cols))


def extraer_reglas_fca(s: str, b: int, fc: Optional[FormalContext] = None) -> dict:
    """Reglas exactas (implicaciones premisa -> y) calculadas sobre el FormalContext.

    Se recorre el árbol de prefijos del contexto llevando la EXTENSIÓN de la premisa
    (extent): extent(B ∪ {c_m=v}) = extent(B) & ext[c_m=v]. Si todos los objetos de esa
    extensión tienen el mismo y, la implicación premisa -> y es exacta y ese nodo ya no
    se sigue expandiendo (sus descendientes serían redundantes, igual que antes).

    Poda por rentabilidad: la frecuencia solo baja y el costo solo sube al profundizar,
    así que un nodo con frecuencia <= costo no puede dar ninguna regla rentable debajo.
    El resultado es idéntico a extraer_reglas_strings(), sin generar subcadenas.
    """
    n = len(s)
    if b < 1 or n - b < 1:
        return {}
    if fc is None:
        fc = construir_contexto_fca(s, b)

    ext = fc.ext
    y1 = ext[2 * b + 1]                                  # objetos con y=1
    base = BITS_PRED + math.ceil(math.log2(b + 1))       # costo(w) = len(w) + base

    reglas = {}
    pila = [("", fc.full_obj)]                           # (prefijo w, extent de su premisa)
    while pila:
        w, e = pila.pop()
        m = len(w)
        for v in (0, 1):
            e2 = e & ext[2 * m + v]
            k = e2.bit_count()                           # frecuencia del prefijo w2
            w2 = w + ("1" if v else "0")
            if k <= len(w2) + base:
                continue                                 # nunca será rentable (ni sus hijos)
            n1 = (e2 & y1).bit_count()
            if n1 == k:                                  # extent ⊆ ext[y=1]
                reglas[w2] = "1"
            elif n1 == 0:                                # extent ⊆ ext[y=0]
                reglas[w2] = "0"
            elif m + 1 < b:
                pila.append((w2, e2))                    # no determinista: seguir bajando

    return dict(sorted(reglas.items(), key=lambda kv: (len(kv[0]), kv[0])))


def extraer_reglas_guigues_duquenne(s: str, b: int) -> dict:
    """Nombre histórico: ahora las reglas se extraen sobre el contexto formal de FCA."""
    return extraer_reglas_fca(s, b)


def extraer_reglas_strings(s: str, b: int) -> dict:
    """Versión anterior (sin FCA, con subcadenas). Solo como referencia para comparar."""
    n = len(s)
    N = n - b - BITS_PRED + 1

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
        if len(c) > 1:
            continue
        if any(w[:k] in deterministicas for k in range(1, len(w))):
            continue

        deterministicas.add(w)
        prediccion = list(c.keys())[0]
        frecuencia = c[prediccion]

        if frecuencia > costo_regla_estimado(w, b):
            reglas[w] = prediccion

    return dict(sorted(reglas.items(), key=lambda kv: (len(kv[0]), kv[0])))


# ----------------------------- verificaciones opcionales --------------------
def _premisa(w: str) -> int:
    B = 0
    for j, c in enumerate(w):
        B |= 1 << (2 * j + int(c))
    return B


def verificar_equivalencia(s: str, b: int) -> bool:
    """Comprueba que la versión FCA y la versión con strings dan las mismas reglas."""
    a = extraer_reglas_fca(s, b)
    r = extraer_reglas_strings(s, b)
    assert a == r, "las reglas FCA y las de strings difieren"
    return True


def validar_reglas_con_cierre(s: str, b: int, reglas: Optional[dict] = None,
                              fc: Optional[FormalContext] = None) -> bool:
    """Comprueba con fc.closure que cada regla es una implicación exacta del contexto."""
    if fc is None:
        fc = construir_contexto_fca(s, b)
    if reglas is None:
        reglas = extraer_reglas_fca(s, b, fc)
    for w, x in reglas.items():
        C = fc.closure(_premisa(w))
        assert (C >> (2 * b + int(x))) & 1, f"regla inválida: {w} -> {x}"
    return True


def validar_con_base_gd(s: str, b: int, reglas: Optional[dict] = None) -> bool:
    """Comprueba que cada regla se deriva de la base de Guigues-Duquenne REAL de tu
    librería (guigues_duquenne(fc)).

    Solo para cadenas cortas (p. ej. "01"*150 con b=8): la base completa sobre las
    posiciones puede tener un número exponencial de implicaciones.
    """
    try:
        from implications.guigues_duquenne import guigues_duquenne
    except ImportError:
        from guigues_duquenne import guigues_duquenne     # si está en otra carpeta
    from implications.armstrong import ImplicationClosure

    fc = construir_contexto_fca(s, b)
    if reglas is None:
        reglas = extraer_reglas_fca(s, b, fc)

    base = guigues_duquenne(fc)
    ic = ImplicationClosure()
    for A, B in base:
        ic.add(A, B)

    for w, x in reglas.items():
        assert (ic(_premisa(w)) >> (2 * b + int(x))) & 1, f"no se deriva de la base GD: {w} -> {x}"

    print(f"Base GD: {len(base)} implicaciones | {len(reglas)} reglas derivadas de ella")
    return True


# ---------------------------------------------------------------------------
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
    reglas = extraer_reglas_fca(s, b)
    semilla, datos = codificar(s, b, reglas)

    # Empaquetamiento de la base de reglas FCA mediante Trie Binario
    trie_root = construir_trie_reglas(reglas)
    trie_bits = serializar_trie(trie_root)
    costo_reglas = len(trie_bits)

    total_bits = len(semilla) + len(datos) + costo_reglas
    return reglas, semilla, datos, costo_reglas, total_bits, trie_bits



# ---------------------------------------------------------------------------
# HÍBRIDO FCA:
#  1) Reglas EXACTAS (implicaciones del FormalContext, extraer_reglas_fca),
#     empaquetadas en el Trie y transmitidas: los bits que deducen no se codifican.
#  2) Implicaciones PARCIALES (Luxenburger): se aprenden de forma incremental durante
#     la codificación/decodificación (nada que transmitir) para los bits no deducidos.
# ---------------------------------------------------------------------------
MODELO_SRC = '''def deducido(contexto, reglas, b):
    for m in range(1, b + 1):
        v = reglas.get(contexto[:m])
        if v is not None:
            return v
    return None


class Modelo:
    def __init__(self, b):
        self.b = b
        self.cnt = {}
        self.conf = {}

    def probar(self, cb):
        cnt = self.cnt
        for m in range(self.b, 0, -1):
            v = cnt.get(cb[:m])
            if v:
                n0 = v & 0xFFFFFFFF
                n1 = v >> 32
                pb = 1 if n1 >= n0 else 0
                mn = n0 if pb else n1
                clave = (m, min(n0 + n1, 5), min(mn, 2))
                st = self.conf.get(clave)
                if st is None:
                    st = self.conf[clave] = [0, 0]
                pr = (st[0] * 2 + 1) * 65536 // (st[1] * 2 + 2)
                pr = min(max(pr, 64), 65472)
                p1 = pr if pb else 65536 - pr
                return 65536 - p1, st, pb
        return 32768, None, 0

    def actualizar(self, cb, bit, st, pb):
        if st is not None:
            st[1] += 1
            if bit == pb:
                st[0] += 1
        cnt = self.cnt
        inc = (1 << 32) if bit else 1
        for m in range(1, self.b + 1):
            k = cb[:m]
            cnt[k] = cnt.get(k, 0) + inc
'''
exec(MODELO_SRC)


def codificar_aritmetico(s: str, b: int, reglas: dict) -> bytes:
    """Codificador de rango binario (estilo LZMA). Los bits deducidos por reglas exactas
    no se codifican; los demás usan las implicaciones parciales aprendidas."""
    n = len(s)
    M = Modelo(b)
    out = bytearray()
    st = {"low": 0, "rng": 0xFFFFFFFF, "cache": 0, "size": 1}

    def shift_low():
        low = st["low"]
        if low < 0xFF000000 or low > 0xFFFFFFFF:
            carry = low >> 32
            temp = st["cache"]
            while True:
                out.append((temp + carry) & 0xFF)
                temp = 0xFF
                st["size"] -= 1
                if st["size"] == 0:
                    break
            st["cache"] = (low >> 24) & 0xFF
        st["size"] += 1
        st["low"] = (low & 0x00FFFFFF) << 8

    for i in range(n - b - 1, -1, -1):
        cb = s[i + 1 : i + 1 + b]
        bit = 1 if s[i] == '1' else 0
        v = deducido(cb, reglas, b)
        if v is not None:
            assert v == s[i], "regla exacta violada"
            M.actualizar(cb, bit, None, 0)
            continue
        p0, est, pb = M.probar(cb)
        bound = (st["rng"] >> 16) * p0
        if bit == 0:
            st["rng"] = bound
        else:
            st["low"] += bound
            st["rng"] -= bound
        while st["rng"] < (1 << 24):
            st["rng"] <<= 8
            shift_low()
        M.actualizar(cb, bit, est, pb)

    low, rng = st["low"], st["rng"]
    hi = low + rng - 1
    for k in range(32, -1, -1):
        v = ((low + (1 << k) - 1) >> k) << k
        if v <= hi:
            st["low"] = v
            break
    for _ in range(5):
        shift_low()

    res = bytearray(out[1:])
    while res and res[-1] == 0:
        res.pop()
    return bytes(res)


def empaquetar(s: str, b: int) -> dict:
    """Blob = semilla + Trie de reglas exactas (alineado a byte) + carga aritmética."""
    n = len(s)
    reglas = extraer_reglas_fca(s, b)
    trie_bits = serializar_trie(construir_trie_reglas(reglas))
    carga = codificar_aritmetico(s, b, reglas)
    cab = s[n - b:] + trie_bits
    pad = (-len(cab)) % 8
    cab += "0" * pad
    blob = int(cab, 2).to_bytes(len(cab) // 8, "big") + carga
    cabecera = math.ceil(math.log2(n + 1)) + max(5, b.bit_length())
    return dict(b=b, blob=blob, reglas=len(reglas), semilla=b, trie=len(trie_bits),
                carga=8 * len(carga), cabecera=cabecera, total=cabecera + 8 * len(blob))


_PLANTILLA_DECODIFICADOR = '''# Decodificador autogenerado: reconstruye la cadena de bits original.
# Metodo: reglas FCA exactas (Trie) + implicaciones parciales aprendidas + codificacion aritmetica.
import base64
import hashlib

N = @@N@@
B = @@B@@
SHA256 = "@@SHA@@"
BLOB = (
@@BLOB@@
)

@@MODELO@@

def reconstruir():
    data = base64.b64decode(BLOB)
    pos = 0

    def leer(k):
        nonlocal pos
        v = 0
        for _ in range(k):
            v = (v << 1) | ((data[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
        return v

    semilla = "".join(str(leer(1)) for _ in range(B))
    reglas = {}

    def trie(w):
        if leer(1) == 1:
            reglas[w] = str(leer(1))
        else:
            h0 = leer(1)
            h1 = leer(1)
            if h0:
                trie(w + "0")
            if h1:
                trie(w + "1")

    trie("")
    carga = data[(pos + 7) >> 3:]
    ip = 0

    def sig():
        nonlocal ip
        v = carga[ip] if ip < len(carga) else 0
        ip += 1
        return v

    code = 0
    for _ in range(4):
        code = (code << 8) | sig()
    rng = 0xFFFFFFFF
    M = Modelo(B)
    res = list(semilla[::-1])
    for _ in range(N - B):
        cb = "".join(res[-1:-B - 1:-1])
        v = deducido(cb, reglas, B)
        if v is not None:
            bit = int(v)
            est = None
            pb = 0
        else:
            p0, est, pb = M.probar(cb)
            bound = (rng >> 16) * p0
            if code < bound:
                rng = bound
                bit = 0
            else:
                code -= bound
                rng -= bound
                bit = 1
            while rng < (1 << 24):
                rng <<= 8
                code = (code << 8) | sig()
        M.actualizar(cb, bit, est, pb)
        res.append("1" if bit else "0")
    return "".join(reversed(res))


def main():
    s = reconstruir()
    ok = len(s) == N and hashlib.sha256(s.encode()).hexdigest() == SHA256
    print("Bits reconstruidos:", len(s), "(esperados:", N, ")")
    print("VERIFICACION SHA-256:", "OK - la cadena se reconstruye tal cual" if ok else "FALLO")


if __name__ == "__main__":
    main()
'''


def generar_decodificador(s: str, info: dict) -> str:
    """Texto de un programa Python autónomo que regenera 's' y verifica su SHA-256."""
    import base64
    import hashlib
    b64 = base64.b64encode(info["blob"]).decode()
    lineas = ['    "' + b64[i:i + 76] + '"' for i in range(0, len(b64), 76)]
    sha = hashlib.sha256(s.encode()).hexdigest()
    return (_PLANTILLA_DECODIFICADOR
            .replace("@@N@@", str(len(s)))
            .replace("@@B@@", str(info["b"]))
            .replace("@@SHA@@", sha)
            .replace("@@BLOB@@", "\n".join(lineas))
            .replace("@@MODELO@@", MODELO_SRC))


def leer_bits(arg: str) -> str:
    """Si arg es la ruta de un archivo, devuelve sus bytes como cadena de 0/1
    (8 bits por byte, MSB primero). Si no, filtra arg como cadena de bits."""
    if os.path.isfile(arg):
        with open(arg, "rb") as f:
            datos = f.read()
        return (np.unpackbits(np.frombuffer(datos, dtype=np.uint8)) + 48).tobytes().decode("ascii")
    return "".join(c for c in arg if c in "01")


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("Uso: python hibrido.py BITS|ARCHIVO [BITS_CONTEXTO]")
    es_archivo = os.path.isfile(sys.argv[1])
    s = leer_bits(sys.argv[1])
    n = len(s)
    if n < 2 + BITS_PRED:
        sys.exit(f"Hacen falta al menos {2 + BITS_PRED} bits.")

    if len(sys.argv) == 3:
        opciones = [int(sys.argv[2])]
    else:
        opciones = [b for b in (12, 16, 20, 24) if b <= n - 1] or [n - 1]

    print(" b | reglas exactas | semilla | Trie | carga aritm. | cabecera | TOTAL (bits) | seg")
    mejor = None
    for b in opciones:
        t0 = time.perf_counter()
        info = empaquetar(s, b)
        dt = time.perf_counter() - t0
        print(f"{b:2d} | {info['reglas']:14d} | {info['semilla']:7d} | {info['trie']:4d} | "
              f"{info['carga']:12d} | {info['cabecera']:8d} | {info['total']} | {dt:.1f}")
        if mejor is None or info["total"] < mejor["total"]:
            mejor = info

    print(f"\nElegido: b = {mejor['b']} | total: {mejor['total']} bits "
          f"({mejor['total'] / n:.4f} del original de {n} bits)")

    codigo = generar_decodificador(s, mejor)
    if es_archivo:
        with open("decodificador.py", "w", encoding="utf-8") as f:
            f.write(codigo)
        print("\nDecodificador guardado en decodificador.py")
        print("Al ejecutarlo debe imprimir: VERIFICACION SHA-256: OK")
    else:
        return 
        print("\n=== DECODIFICADOR GENERADO (copie desde INICIO hasta FIN, sin las marcas) ===")
        print("########## INICIO DEL CÓDIGO ##########")
        print(codigo)
        print("########## FIN DEL CÓDIGO ##########")
        print("Al ejecutarlo debe imprimir: VERIFICACION SHA-256: OK")


if __name__ == "__main__":
    main()
