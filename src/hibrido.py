import sys
import os
import math
import time
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

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
            bits_out.append('1' if '*' in n.children else '0')
            if '0' in n.children:
                dfs(n.children['0'])
            if '1' in n.children:
                dfs(n.children['1'])
            if '*' in n.children:
                dfs(n.children['*'])

    dfs(root)
    return "".join(bits_out)


def deserializar_trie(bitstr: str) -> dict:
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
            has_any = bitstr[pos + 2]
            pos += 3
            if has_0 == '1':
                dfs(current_w + '0')
            if has_1 == '1':
                dfs(current_w + '1')
            if has_any == '1':
                dfs(current_w + '*')

    dfs("")
    return reglas


def costo_regla_estimado(w: str, b: int) -> int:
    return len(w) + BITS_PRED + math.ceil(math.log2(b + 1))


def construir_contexto_fca(s: str, b: int) -> FormalContext:
    assert BITS_PRED == 1, "el contexto FCA asume BITS_PRED = 1"
    n = len(s)
    N = n - b
    arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8) == ord("1")
    W = sliding_window_view(arr, b + 1)

    M = np.empty((N, 2 * b + 2), dtype=bool)
    ctx = W[:, 1:]
    M[:, 0:2 * b:2] = ~ctx
    M[:, 1:2 * b:2] = ctx
    M[:, 2 * b] = ~W[:, 0]
    M[:, 2 * b + 1] = W[:, 0]

    cols = [f"c{j}={v}" for j in range(b) for v in (0, 1)] + ["y=0", "y=1"]
    return FormalContext(pd.DataFrame(M, columns=cols))


def extraer_reglas_prefijos(s: str, b: int, fc: Optional[FormalContext] = None) -> dict:
    n = len(s)
    if b < 1 or n - b < 1:
        return {}
    if fc is None:
        fc = construir_contexto_fca(s, b)

    ext = fc.ext
    y1 = ext[2 * b + 1]
    base = BITS_PRED + math.ceil(math.log2(b + 1))

    reglas = {}
    pila = [("", fc.full_obj)]
    while pila:
        w, e = pila.pop()
        m = len(w)
        for v in (0, 1):
            e2 = e & ext[2 * m + v]
            k = e2.bit_count()
            w2 = w + ("1" if v else "0")
            if k <= len(w2) + base:
                continue
            n1 = (e2 & y1).bit_count()
            if n1 == k:
                reglas[w2] = "1"
            elif n1 == 0:
                reglas[w2] = "0"
            elif m + 1 < b:
                pila.append((w2, e2))

    return dict(sorted(reglas.items(), key=lambda kv: (len(kv[0]), kv[0])))


def extraer_reglas_fca(s: str, b: int, fc: Optional[FormalContext] = None) -> dict:
    if b < 1 or len(s) <= b:
        return {}
    if fc is None:
        fc = construir_contexto_fca(s, b)

    reglas = extraer_reglas_prefijos(s, b, fc)
    y1 = fc.ext[2 * b + 1]
    frontera = [((), fc.full_obj, 0)]

    for profundidad in range(min(b, 3) + 1):
        siguientes = []
        for premisa, e, inicio in frontera:
            k = e.bit_count()
            n1 = (e & y1).bit_count()
            if n1 == 0 or n1 == k:
                ancho = premisa[-1][0] + 1 if premisa else 0
                w = ['*'] * ancho
                for j, v in premisa:
                    w[j] = str(v)
                reglas[''.join(w)] = '1' if n1 else '0'
                continue
            if profundidad == min(b, 3):
                continue
            for j in range(inicio, b):
                for v in (0, 1):
                    e2 = e & fc.ext[2 * j + v]
                    if e2:
                        siguientes.append((premisa + ((j, v),), e2, j + 1))

        def prioridad(item):
            premisa, e, _ = item
            k = e.bit_count()
            n1 = (e & y1).bit_count()
            return (min(n1, k - n1) / k, -k, premisa)

        siguientes.sort(key=prioridad)
        frontera = siguientes[:64]

    return dict(sorted(reglas.items(), key=lambda kv: (len(kv[0]), kv[0])))


MODELO_SRC = '''def deducido(contexto, reglas, b):
    for w, v in reglas.items():
        if all(bit == '*' or bit == contexto[j] for j, bit in enumerate(w)):
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


def evaluar_reglas(s: str, b: int, reglas: dict) -> dict:
    n = len(s)
    trie_bits = serializar_trie(construir_trie_reglas(reglas))
    carga = codificar_aritmetico(s, b, reglas)
    cab = s[n - b:] + trie_bits
    pad = (-len(cab)) % 8
    cab += "0" * pad
    blob = int(cab, 2).to_bytes(len(cab) // 8, "big") + carga
    cabecera = math.ceil(math.log2(n + 1)) + max(5, b.bit_length())
    return dict(b=b, blob=blob, reglas=len(reglas), semilla=b, trie=len(trie_bits),
                carga=8 * len(carga), cabecera=cabecera, total=cabecera + 8 * len(blob))


def seleccionar_reglas(s: str, b: int, candidatos: dict) -> tuple[dict, dict]:
    reglas = {}
    mejor = evaluar_reglas(s, b, reglas)
    prefijos = extraer_reglas_prefijos(s, b)
    info = evaluar_reglas(s, b, prefijos)
    if info['total'] < mejor['total']:
        reglas, mejor = prefijos, info

    fc = construir_contexto_fca(s, b)
    costos = [0.0] * (len(s) - b)
    modelo = Modelo(b)
    for i in range(len(s) - b - 1, -1, -1):
        cb = s[i + 1:i + 1 + b]
        bit = int(s[i])
        est, pb = None, 0
        if deducido(cb, reglas, b) is None:
            p0, est, pb = modelo.probar(cb)
            prob = p0 if bit == 0 else 65536 - p0
            costos[i] = -math.log2(prob / 65536)
        modelo.actualizar(cb, bit, est, pb)

    def prioridad(w):
        e = fc.full_obj
        for j, bit in enumerate(w):
            if bit != '*':
                e &= fc.ext[2 * j + int(bit)]
        costo = len(serializar_trie(construir_trie_reglas({w: candidatos[w]})))
        ahorro = 0.0
        while e:
            bit = e & -e
            ahorro += costos[bit.bit_length() - 1]
            e ^= bit
        return ahorro / costo

    pendientes = sorted(candidatos, key=prioridad, reverse=True)[:32]
    while pendientes:
        elegida = None
        for w in pendientes:
            if w in reglas:
                continue
            prueba = dict(reglas)
            prueba[w] = candidatos[w]
            info = evaluar_reglas(s, b, prueba)
            if info['total'] < mejor['total']:
                elegida = w
                nuevas, mejor = prueba, info
        if elegida is None:
            break
        reglas = nuevas
        pendientes.remove(elegida)

    cambio = True
    while cambio:
        cambio = False
        for w in list(reglas):
            prueba = {k: v for k, v in reglas.items() if k != w}
            info = evaluar_reglas(s, b, prueba)
            if info['total'] < mejor['total']:
                reglas, mejor = prueba, info
                cambio = True

    return reglas, mejor


def empaquetar(s: str, b: int) -> dict:
    if not 1 <= b < len(s):
        raise ValueError("El contexto debe estar entre 1 y len(s) - 1")
    candidatos = extraer_reglas_fca(s, b)
    _, info = seleccionar_reglas(s, b, candidatos)
    return info


def _evaluar_worker(args):
    """Worker global para paralelizar en múltiples núcleos."""
    s, b = args
    t0 = time.perf_counter()
    info = empaquetar(s, b)
    dt = time.perf_counter() - t0
    info["dt"] = dt
    return info


_PLANTILLA_DECODIFICADOR = '''# Decodificador autogenerado: reconstruye la cadena de bits original.
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
            h_any = leer(1)
            if h0:
                trie(w + "0")
            if h1:
                trie(w + "1")
            if h_any:
                trie(w + "*")

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


def extraer_texto_pdf(ruta: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            sys.exit("Error: Instale 'pypdf' para extraer texto de archivos PDF (pip install pypdf).")

    reader = PdfReader(ruta)
    texto = "".join(page.extract_text() or "" for page in reader.pages)
    return texto


def leer_bits(arg: str) -> str:
    if os.path.isfile(arg):
        if arg.lower().endswith(".pdf"):
            texto = extraer_texto_pdf(arg)
            print(f"[PDF] Texto extraído: {len(texto)} caracteres")
            datos = texto.encode("utf-8")
        else:
            with open(arg, "rb") as f:
                datos = f.read()
        return (np.unpackbits(np.frombuffer(datos, dtype=np.uint8)) + 48).tobytes().decode("ascii")
    return "".join(c for c in arg if c in "01")


def obtener_rango_optimo(n: int) -> list[int]:
    """Calcula la ventana de candidatos óptima b basada en log2(N)."""
    log_n = int(math.log2(n))
    b_min = max(4, log_n - 2)
    b_max = min(n - 1, log_n + 16)
    
    # Muestreo con paso 4 dentro de la cota logarítmica
    rango = list(range(b_min, b_max + 1, 4))
    if not rango:
        rango = [max(2, n - 1)]
    return rango


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
        opciones = obtener_rango_optimo(n)

    print(f"Búsqueda por Rango Inteligente (log2 N = {int(math.log2(n))}): evaluando {len(opciones)} candidatos {opciones}")
    print(" b | reglas exactas | semilla | Trie | carga aritm. | cabecera | TOTAL (bits) | seg")
    
    mejor = None
    
    # Paralelización utilizando los núcleos de la CPU
    with ProcessPoolExecutor() as executor:
        resultados = executor.map(_evaluar_worker, [(s, b) for b in opciones])
        for info in resultados:
            print(f"{info['b']:2d} | {info['reglas']:14d} | {info['semilla']:7d} | {info['trie']:4d} | "
                  f"{info['carga']:12d} | {info['cabecera']:8d} | {info['total']} | {info['dt']:.2f}")
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


if __name__ == "__main__":
    main()
