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



# ---------------------------------------------------------------------------
# HÍBRIDO FCA:
#  1) Reglas EXACTAS (base de Guigues-Duquenne, extraer_reglas_guigues_duquenne),
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
    reglas = extraer_reglas_guigues_duquenne(s, b)
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
        opciones = [b for b in (12, 16, 20, 24) if b <= n - 1] or [n - 1]

    print(" b | reglas exactas | semilla | Trie | carga aritm. | cabecera | TOTAL (bits)")
    mejor = None
    for b in opciones:
        info = empaquetar(s, b)
        print(f"{b:2d} | {info['reglas']:14d} | {info['semilla']:7d} | {info['trie']:4d} | "
              f"{info['carga']:12d} | {info['cabecera']:8d} | {info['total']}")
        if mejor is None or info["total"] < mejor["total"]:
            mejor = info

    print(f"\nElegido: b = {mejor['b']} | total: {mejor['total']} bits "
          f"({mejor['total'] / n:.4f} del original de {n} bits)")

    codigo = generar_decodificador(s, mejor)
    print("\n=== DECODIFICADOR GENERADO (copie desde INICIO hasta FIN, sin las marcas) ===")
    print("########## INICIO DEL CÓDIGO ##########")
    print(codigo)
    print("########## FIN DEL CÓDIGO ##########")
    print("Al ejecutarlo debe imprimir: VERIFICACION SHA-256: OK")


if __name__ == "__main__":
    main()
