import sys
import math
from fca.context import FormalContext, bits
import os
import numpy as np


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



# ---------------------------------------------------------------------------
# COMPRESIÓN ADAPTATIVA: las implicaciones FCA (exactas y parciales) se aprenden
# mientras se codifica/decodifica. No se transmite árbol ni reglas.
# Contexto = siguientes m bits (m = 1..b); se usa el más largo ya visto, su bit
# mayoritario es la "conclusión" y la confianza de esa implicación también se aprende.
# ---------------------------------------------------------------------------
MODELO_SRC = '''class Modelo:
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


def codificar_aritmetico(s: str, b: int) -> bytes:
    """Codificador de rango binario (estilo LZMA); probabilidades del modelo adaptativo."""
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
        p0, est, pb = M.probar(cb)
        bit = 1 if s[i] == '1' else 0
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
    """Blob = semilla (b bits, alineada a byte) + carga aritmética."""
    n = len(s)
    carga = codificar_aritmetico(s, b)
    cab = s[n - b:]
    pad = (-len(cab)) % 8
    cab += "0" * pad
    blob = int(cab, 2).to_bytes(len(cab) // 8, "big") + carga
    cabecera = math.ceil(math.log2(n + 1)) + max(5, b.bit_length())
    return dict(b=b, blob=blob, semilla=b, carga=8 * len(carga), cabecera=cabecera,
                total=cabecera + 8 * len(blob))


_PLANTILLA_DECODIFICADOR = '''# Decodificador autogenerado: reconstruye la cadena de bits original.
# Metodo: implicaciones aprendidas de forma incremental + codificacion aritmetica.
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
    nb = (B + 7) // 8
    semilla = "".join(format(x, "08b") for x in data[:nb])[:B]
    carga = data[nb:]
    pos = 0

    def sig():
        nonlocal pos
        v = carga[pos] if pos < len(carga) else 0
        pos += 1
        return v

    code = 0
    for _ in range(4):
        code = (code << 8) | sig()
    rng = 0xFFFFFFFF
    M = Modelo(B)
    res = list(semilla[::-1])
    for _ in range(N - B):
        cb = "".join(res[-1:-B - 1:-1])
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


def cabecera_ref(n: int) -> int:
    return math.ceil(math.log2(n + 1)) + 5


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

def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("Uso: python modelo_rapido.py BITS|ARCHIVO [BITS_CONTEXTO]")
    
    es_archivo = os.path.isfile(sys.argv[1])
    s = leer_bits(sys.argv[1])
    n = len(s)
    if n < 2 + BITS_PRED:
        sys.exit(f"Hacen falta al menos {2 + BITS_PRED} bits.")

    if len(sys.argv) == 3:
        opciones = [int(sys.argv[2])]
    else:
        opciones = [b for b in range(12, 68, 4) if b <= n - 1] or [n - 1]

    print(" b | semilla | carga aritm. | cabecera | TOTAL (bits)")
    mejor = None
    for b in opciones:
        info = empaquetar(s, b)
        print(f"{b:2d} | {info['semilla']:7d} | {info['carga']:12d} | "
              f"{info['cabecera']:8d} | {info['total']}")
        if mejor is None or info["total"] < mejor["total"]:
            mejor = info

    b = mejor["b"]
    print(f"\nElegido: b = {b} | total: {mejor['total']} bits "
          f"({mejor['total'] / n:.4f} del original {n} bits)")

    codigo = generar_decodificador(s, mejor)
    if es_archivo:
        with open("decodificador.py", "w", encoding="utf-8") as f:
            f.write(codigo)
        print("\nDecodificador guardado en decodificador.py")
        print("Al ejecutarlo debe imprimir: VERIFICACION SHA-256: OK")


if __name__ == "__main__":
    main()
