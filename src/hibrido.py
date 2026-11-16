"""Compresión sin pérdida con reglas FCA y un residuo aritmético.

La selección usa MDL: cabecera + semilla + trie + relleno + residuo.
El contenedor autodelimitado z y el decodificador fijo D satisfacen D(z)=s;
por tanto K_U(s) <= 8*len(z) + c_D. No se calcula K ni se garantiza el
óptimo: la búsqueda de reglas y contextos es heurística.
"""

import sys
import os
import math
import time
from typing import Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from heapq import nsmallest, nlargest

import numpy as np

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


@dataclass(slots=True)
class ContextoBits:
    """Columnas FCA; evita materializar las filas que el compresor no utiliza."""
    ext: list[int]
    full_obj: int


def construir_contexto_fca(s: str, b: int) -> ContextoBits:
    assert BITS_PRED == 1, "el contexto FCA asume BITS_PRED = 1"
    if not 1 <= b < len(s):
        raise ValueError("El contexto debe estar entre 1 y len(s) - 1")
    full = (1 << (len(s) - b)) - 1
    cadena = int(s[::-1], 2)  # el objeto i ocupa el bit i de cada extensión
    ext = []
    for j in range(1, b + 1):
        unos = (cadena >> j) & full
        ext.extend((full ^ unos, unos))
    unos = cadena & full
    ext.extend((full ^ unos, unos))
    return ContextoBits(ext, full)


def extraer_reglas_prefijos(s: str, b: int, fc: Optional[ContextoBits] = None) -> dict:
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


def extraer_reglas_fca(s: str, b: int, fc: Optional[ContextoBits] = None,
                      prefijos=None) -> dict:
    if b < 1 or len(s) <= b:
        return {}
    if fc is None:
        fc = construir_contexto_fca(s, b)

    reglas = dict(extraer_reglas_prefijos(s, b, fc) if prefijos is None else prefijos)
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

        frontera = nsmallest(64, siguientes, key=prioridad)

    return dict(sorted(reglas.items(), key=lambda kv: (len(kv[0]), kv[0])))


MODELO_SRC = '''def deducido(contexto, reglas, b):
    for w, v in reglas.items():
        if all(bit == '*' or bit == contexto[j] for j, bit in enumerate(w)):
            return v
    return None


class Modelo:
    def __init__(self, b):
        self.b = b
        self.hijos = [[0, 0]]
        self.cuentas = [0]
        self.rutas = {}
        self.ultimo = None
        self.ruta_actual = None
        self.conf = {}

    def ruta(self, cb):
        if cb == self.ultimo:
            return self.ruta_actual
        ruta = self.rutas.get(cb)
        if ruta is None:
            ruta = []
            nodo = 0
            for bit in cb:
                rama = bit == '1'
                siguiente = self.hijos[nodo][rama]
                if siguiente == 0:
                    siguiente = len(self.cuentas)
                    self.hijos[nodo][rama] = siguiente
                    self.hijos.append([0, 0])
                    self.cuentas.append(0)
                nodo = siguiente
                ruta.append(nodo)
            if len(self.rutas) < 512:
                self.rutas[cb] = ruta
        self.ultimo, self.ruta_actual = cb, ruta
        return ruta

    def probar(self, cb):
        ruta = self.ruta(cb)
        for m in range(len(ruta), 0, -1):
            v = self.cuentas[ruta[m - 1]]
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
        inc = (1 << 32) if bit else 1
        cuentas = self.cuentas
        for nodo in self.ruta(cb):
            cuentas[nodo] += inc
'''
exec(MODELO_SRC)


def preparar_modelo(s: str, b: int) -> list:
    """Mismas estadísticas causales que Modelo, calculadas por grupos en NumPy.

    El orden estable conserva el recorrido de derecha a izquierda. Cada
    predicción solo cuenta bits ya reconstruidos, nunca el bit actual.
    Las ventanas grandes conservan la ruta de enteros Python sin truncamiento.
    """
    if b > 64 or len(s) - b < 256:
        return _preparar_modelo_trie(s, b)
    n = len(s) - b
    arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8) & 1
    contextos = np.zeros(n, dtype=np.uint64)
    for j in range(1, b + 1):
        contextos <<= np.uint64(1)
        contextos |= arr[j:j + n]
    contextos = contextos[::-1].copy()
    valores = arr[:n][::-1].astype(np.int64)
    claves = np.full(n, -1, dtype=np.int64)
    pred = np.zeros(n, dtype=np.int64)
    posiciones = np.arange(n)
    pendientes = n
    for m in range(b, 0, -1):
        ctx = contextos[:pendientes] >> np.uint64(b - m)
        orden = np.argsort(ctx, kind='stable')
        grupos = ctx[orden]
        inicios = np.empty(pendientes, dtype=bool)
        inicios[0] = True
        inicios[1:] = grupos[1:] != grupos[:-1]
        bases = np.maximum.accumulate(np.where(inicios, posiciones[:pendientes], 0))
        total = posiciones[:pendientes] - bases
        unos = valores[orden]
        previos = np.cumsum(unos) - unos
        n1 = previos - previos[bases]
        tomar = (total > 0) & (claves[orden] < 0)
        ids = orden[tomar]
        n1 = n1[tomar]
        n0 = total[tomar] - n1
        pred[ids] = n1 >= n0
        claves[ids] = (m * 18 + np.minimum(total[tomar], 5) * 3
                       + np.minimum(np.minimum(n0, n1), 2))
        # Los eventos posteriores al último pendiente ya tienen su contexto
        # más largo y no pueden influir en predicciones anteriores.
        pendientes = int(np.flatnonzero(claves[:pendientes] < 0)[-1]) + 1
        if pendientes == 1:
            break
    return list(zip(valores[::-1].tolist(),
                    [None if c < 0 else c for c in claves[::-1].tolist()],
                    pred[::-1].tolist()))


def _preparar_modelo_trie(s: str, b: int) -> list:
    hijos = [[0, 0]]
    cuentas = [0]
    rutas = {}
    datos = [None] * (len(s) - b)
    cadena = s.encode('ascii')

    for i in range(len(datos) - 1, -1, -1):
        bit = cadena[i] & 1
        inc = (1 << 32) if bit else 1
        cb = cadena[i + 1:i + b + 1]
        ruta = rutas.get(cb)
        if ruta is not None:
            largo, vistos = b, cuentas[ruta[-1]]
            for nodo in ruta:
                cuentas[nodo] += inc
        else:
            ruta = [] if len(rutas) < 512 else None
            nodo, largo, vistos = 0, 0, 0
            for m, v in enumerate(cb, 1):
                rama = v & 1
                siguiente = hijos[nodo][rama]
                if siguiente == 0:
                    siguiente = len(cuentas)
                    hijos[nodo][rama] = siguiente
                    hijos.append([0, 0])
                    cuentas.append(0)
                nodo = siguiente
                cuenta = cuentas[nodo]
                if cuenta:
                    largo, vistos = m, cuenta
                cuentas[nodo] = cuenta + inc
                if ruta is not None:
                    ruta.append(nodo)
            if ruta is not None:
                rutas[cb] = ruta

        clave, pb = None, 0
        if largo:
            n0 = vistos & 0xFFFFFFFF
            n1 = vistos >> 32
            pb = int(n1 >= n0)
            mn = n0 if pb else n1
            clave = largo * 18 + min(n0 + n1, 5) * 3 + min(mn, 2)
        datos[i] = bit, clave, pb

    return datos


def extension_regla(w: str, fc: ContextoBits) -> int:
    e = fc.full_obj
    for j, bit in enumerate(w):
        if bit != '*':
            e &= fc.ext[2 * j + int(bit)]
    return e


def probabilidad_modelo(clave, pb: int, conf: dict):
    if clave is None:
        return 32768, None
    est = conf.get(clave)
    if est is None:
        est = conf[clave] = [0, 0]
    pr = (est[0] * 2 + 1) * 65536 // (est[1] * 2 + 2)
    pr = min(max(pr, 64), 65472)
    return (65536 - pr if pb else pr), est


def preparar_eventos(datos):
    eventos = np.array([(bit, -1 if clave is None else clave, pb)
                        for bit, clave, pb in reversed(datos)], dtype=np.int64)
    orden = np.argsort(eventos[:, 1], kind='stable')
    return eventos, orden


def probabilidades_residuo(eventos, orden, cobertura):
    n = len(eventos)
    if cobertura:
        marcas = np.frombuffer(cobertura.to_bytes((n + 7) // 8, 'little'), dtype=np.uint8)
        disponibles = np.unpackbits(marcas, bitorder='little', count=n)[::-1] == 0
        orden = orden[disponibles[orden]]
    else:
        disponibles = slice(None)
    if not len(orden):
        return iter(())

    valores = eventos[orden]
    claves = valores[:, 1]
    inicios = np.empty(len(orden), dtype=bool)
    inicios[0] = True
    inicios[1:] = claves[1:] != claves[:-1]
    posiciones = np.arange(len(orden))
    bases = np.maximum.accumulate(np.where(inicios, posiciones, 0))
    aciertos = (valores[:, 0] == valores[:, 2]).astype(np.int64)
    previos = np.cumsum(aciertos) - aciertos
    exitos = previos - previos[bases]
    total = posiciones - bases
    pr = np.clip((exitos * 2 + 1) * 65536 // (total * 2 + 2), 64, 65472)
    p0 = np.where(valores[:, 2], 65536 - pr, pr)
    p0[claves < 0] = 32768
    probabilidades = np.empty(n, dtype=np.int64)
    probabilidades[orden] = p0
    return zip(eventos[disponibles, 0].tolist(), probabilidades[disponibles].tolist())


def probabilidades_secuenciales(datos, b: int, cobertura: int):
    marcas = format(cobertura, f'0{len(datos)}b') if cobertura else None
    conf = [None] * (18 * (b + 1))
    for i, (bit, clave, pb) in enumerate(reversed(datos)):
        if marcas is not None and marcas[i] == '1':
            continue
        p0 = 32768
        if clave is not None:
            est = conf[clave]
            if est is None:
                est = conf[clave] = [0, 0]
            pr = (est[0] * 2 + 1) * 65536 // (est[1] * 2 + 2)
            if pr < 64:
                pr = 64
            elif pr > 65472:
                pr = 65472
            p0 = 65536 - pr if pb else pr
            est[1] += 1
            est[0] += bit == pb
        yield bit, p0


def codificar_aritmetico(s: str, b: int, reglas: dict,
                         datos=None, cobertura=None, limite=None, eventos=None) -> Optional[bytes]:
    n = len(s)
    if datos is None:
        datos = preparar_modelo(s, b)
    if cobertura is None:
        fc = construir_contexto_fca(s, b)
        cobertura = 0
        for w, valor in reglas.items():
            e = extension_regla(w, fc)
            assert not e & fc.ext[2 * b + 1 - int(valor)], "regla exacta violada"
            cobertura |= e
    if cobertura == (1 << (n - b)) - 1:
        return b''
    if eventos is None and len(datos) >= 256:
        eventos = preparar_eventos(datos)
    if eventos is not None:
        pendientes = probabilidades_residuo(*eventos, cobertura)
    else:
        pendientes = probabilidades_secuenciales(datos, b, cobertura)

    out = bytearray()
    low, rng, cache, size, ultimo = 0, 0xFFFFFFFF, 0, 1, 0

    def shift_low():
        nonlocal low, cache, size, ultimo
        if low < 0xFF000000 or low > 0xFFFFFFFF:
            carry = low >> 32
            temp = cache
            while True:
                valor = (temp + carry) & 0xFF
                out.append(valor)
                if valor:
                    ultimo = len(out) - 1
                temp = 0xFF
                size -= 1
                if size == 0:
                    break
            cache = (low >> 24) & 0xFF
        size += 1
        low = (low & 0x00FFFFFF) << 8

    for bit, p0 in pendientes:
        bound = (rng >> 16) * p0
        if bit == 0:
            rng = bound
        else:
            low += bound
            rng -= bound
        while rng < (1 << 24):
            rng <<= 8
            shift_low()
        if limite is not None and ultimo >= limite:
            return None

    hi = low + rng - 1
    for k in range(32, -1, -1):
        v = ((low + (1 << k) - 1) >> k) << k
        if v <= hi:
            low = v
            break
    for _ in range(5):
        shift_low()

    res = bytearray(out[1:])
    while res and res[-1] == 0:
        res.pop()
    return bytes(res)


MAGIC = b'HF\x01'


def _entero_variable(n: int) -> bytes:
    """LEB128 sin signo: termina en el primer byte con bit alto cero."""
    if n < 0:
        raise ValueError("Se requiere un entero no negativo")
    out = bytearray()
    while n >= 128:
        out.append((n & 127) | 128)
        n >>= 7
    out.append(n)
    return bytes(out)


def _cabecera(n: int, b: int, tamano: int) -> bytes:
    return MAGIC + _entero_variable(n) + _entero_variable(b) + _entero_variable(tamano)


def _informacion(n, b, reglas, trie, cab, carga):
    blob = int(cab, 2).to_bytes(len(cab) // 8, 'big') + carga
    cabecera = 8 * len(_cabecera(n, b, len(blob)))
    return dict(n=n, b=b, blob=blob, reglas=len(reglas), semilla=b, trie=len(trie),
                relleno=len(cab) - b - len(trie), carga=8 * len(carga),
                cabecera=cabecera, total=cabecera + 8 * len(blob))


def serializar_paquete(info: dict) -> bytes:
    """Descripción completa: versión, n, b, longitud del blob y blob.

    Su longitud coincide exactamente con info['total']/8. El tamaño explícito
    hace autodelimitado el formato, incluso si la carga termina en ceros.
    """
    blob = info['blob']
    return _cabecera(info['n'], info['b'], len(blob)) + blob


def desempaquetar(paquete: bytes) -> str:
    if not paquete.startswith(MAGIC):
        raise ValueError("Formato o versión de paquete incorrectos")
    pos = len(MAGIC)

    def entero():
        nonlocal pos
        valor = 0
        for desplazamiento in range(0, 70, 7):
            if pos >= len(paquete):
                raise ValueError("Cabecera truncada")
            byte = paquete[pos]
            pos += 1
            valor |= (byte & 127) << desplazamiento
            if byte < 128:
                if desplazamiento and byte == 0:
                    raise ValueError("Entero no canónico")
                return valor
        raise ValueError("Entero demasiado largo")

    n, b, tamano = entero(), entero(), entero()
    if not 1 <= b < n or tamano != len(paquete) - pos:
        raise ValueError("Longitudes del paquete inválidas")
    if 8 * tamano < b + 2:
        raise ValueError("Semilla o trie truncados")
    try:
        return reconstruir_datos(paquete[pos:], n, b)
    except (IndexError, RecursionError) as exc:
        raise ValueError("Trie truncado o inválido") from exc


def evaluar_reglas(s: str, b: int, reglas: dict, datos=None, cobertura=None) -> dict:
    n = len(s)
    trie_bits = serializar_trie(construir_trie_reglas(reglas))
    carga = codificar_aritmetico(s, b, reglas, datos, cobertura)
    cab = s[n - b:] + trie_bits
    pad = (-len(cab)) % 8
    cab += "0" * pad
    return _informacion(n, b, reglas, trie_bits, cab, carga)


def seleccionar_reglas(s: str, b: int, candidatos: dict,
                       fc: Optional[ContextoBits] = None, *, prefijos=None,
                       max_candidatos=32) -> tuple[dict, dict]:
    if fc is None:
        fc = construir_contexto_fca(s, b)
    datos = preparar_modelo(s, b)
    eventos = preparar_eventos(datos) if len(datos) >= 256 else None
    extensiones = {}
    evaluaciones = {}
    cargas = {}

    def evaluar(reglas, limite=None):
        clave = tuple(sorted(reglas.items()))
        if clave in evaluaciones:
            return evaluaciones[clave]
        trie_bits = serializar_trie(construir_trie_reglas(reglas))
        cab = s[len(s) - b:] + trie_bits
        cab += '0' * (-len(cab) % 8)
        # Cota inferior: la longitud del blob (LEB128) puede crecer al añadir
        # carga. Usarla para podar nunca descarta una representación mejor.
        costo = 8 * len(_cabecera(len(s), b, len(cab) // 8)) + len(cab)
        if limite is not None and costo >= limite:
            return None

        cobertura = 0
        for w, valor in reglas.items():
            if w not in extensiones:
                extensiones[w] = extension_regla(w, fc)
            e = extensiones[w]
            assert not e & fc.ext[2 * b + 1 - int(valor)], "regla exacta violada"
            cobertura |= e

        carga = cargas.get(cobertura)
        if carga is None:
            tope = (limite - costo + 7) // 8 if limite is not None else None
            carga = codificar_aritmetico(s, b, reglas, datos, cobertura, tope, eventos)
            if carga is None:
                return None
            if len(cargas) >= 32:
                cargas.pop(next(iter(cargas)))
            cargas[cobertura] = carga

        info = _informacion(len(s), b, reglas, trie_bits, cab, carga)
        evaluaciones[clave] = info
        return info

    reglas = {}
    mejor = evaluar(reglas)
    if not candidatos:
        return reglas, mejor
    if prefijos is None:
        prefijos = extraer_reglas_prefijos(s, b, fc)
    info = evaluar(prefijos)
    if info['total'] < mejor['total']:
        reglas, mejor = prefijos, info

    cobertura = 0
    for w in reglas:
        cobertura |= extensiones[w]

    def mascara(e):
        packed = np.frombuffer(e.to_bytes((len(datos) + 7) // 8, 'little'), dtype=np.uint8)
        return np.unpackbits(packed, bitorder='little', count=len(datos)).astype(bool)

    costos = np.zeros(len(datos))
    if cobertura != fc.full_obj:
        probabilidades = (probabilidades_residuo(*eventos, cobertura) if eventos is not None
                          else probabilidades_secuenciales(datos, b, cobertura))
        probs = np.fromiter((p0 if bit == 0 else 65536 - p0
                             for bit, p0 in probabilidades), dtype=np.float64)
        costos[~mascara(cobertura)] = -np.log2(probs[::-1] / 65536)

    def prioridad(w):
        if w not in extensiones:
            extensiones[w] = extension_regla(w, fc)
        costo = 4 * len(w) + 2
        ahorro = costos[mascara(extensiones[w])].sum() if cobertura != fc.full_obj else 0.0
        return ahorro / costo

    pendientes = nlargest(max_candidatos, candidatos, key=prioridad)
    while pendientes:
        elegida = None
        for w in pendientes:
            if w in reglas:
                continue
            prueba = dict(reglas)
            prueba[w] = candidatos[w]
            info = evaluar(prueba, mejor['total'])
            if info is not None and info['total'] < mejor['total']:
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
            info = evaluar(prueba, mejor['total'])
            if info is not None and info['total'] < mejor['total']:
                reglas, mejor = prueba, info
                cambio = True

    return reglas, mejor


def empaquetar(s: str, b: int, *, modo='completo') -> dict:
    """Busca una descripción MDL para un contexto fijo.

    rapido limita a cuatro candidatos adicionales; completo considera 32.
    Ambos comparan también el modelo sin reglas y el conjunto de prefijos.
    La API conserva completo por defecto; la CLI prioriza rapido.
    """
    if not 1 <= b < len(s):
        raise ValueError("El contexto debe estar entre 1 y len(s) - 1")
    if s.strip('01'):
        raise ValueError("La cadena debe contener únicamente ceros y unos")
    if modo not in ('rapido', 'completo'):
        raise ValueError("Modo desconocido: use rapido o completo")
    fc = construir_contexto_fca(s, b)
    prefijos = extraer_reglas_prefijos(s, b, fc)
    candidatos = extraer_reglas_fca(s, b, fc, prefijos)
    _, info = seleccionar_reglas(s, b, candidatos, fc, prefijos=prefijos,
                                max_candidatos=4 if modo == 'rapido' else 32)
    return info


def _evaluar_worker(args):
    """Worker global para paralelizar en múltiples núcleos."""
    s, b, modo = args
    t0 = time.perf_counter()
    info = empaquetar(s, b, modo=modo)
    dt = time.perf_counter() - t0
    info["dt"] = dt
    return info


_DECODIFICADOR_SRC = '''def reconstruir_datos(data, N, B):
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
    compiladas = []
    for w, valor in reglas.items():
        mascara, patron = 0, 0
        for j, bit in enumerate(w):
            if bit != '*':
                mascara |= 1 << (B - j - 1)
                patron |= int(bit) << (B - j - 1)
        compiladas.append((mascara, patron, valor))
    cache_reglas = {}
    ciclos = {}
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
    cb = semilla
    contexto = int(semilla, 2)
    for _ in range(N - B):
        v = None
        if compiladas:
            v = cache_reglas.get(contexto, -1)
            if v == -1:
                v = None
                for mascara, patron, valor in compiladas:
                    if contexto & mascara == patron:
                        v = valor
                        break
                if len(cache_reglas) >= 4096:
                    cache_reglas.clear()
                cache_reglas[contexto] = v
        if v is not None:
            # Un ciclo formado exclusivamente por reglas exactas se repetirá
            # hasta N: ninguna predicción aritmética puede intervenir después.
            inicio = ciclos.get(contexto)
            if inicio is not None:
                patron = res[inicio:]
                vueltas, resto = divmod(N - len(res), len(patron))
                res.extend(patron * vueltas)
                res.extend(patron[:resto])
                break
            if len(ciclos) >= 4096:
                ciclos.clear()
            ciclos[contexto] = len(res)
            bit = int(v)
            est = None
            pb = 0
        else:
            ciclos.clear()
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
        valor = "1" if bit else "0"
        res.append(valor)
        cb = valor + cb[:-1]
        contexto = (bit << (B - 1)) | (contexto >> 1)
    return "".join(reversed(res))
'''
exec(_DECODIFICADOR_SRC)


_PLANTILLA_DECODIFICADOR = '''# Decodificador autogenerado: reconstruye la cadena de bits original.
import base64
import hashlib
import sys

N = @@N@@
B = @@B@@
SHA256 = "@@SHA@@"
BLOB = (
@@BLOB@@
)

@@MODELO@@
@@DECODIFICADOR@@

def reconstruir():
    return reconstruir_datos(base64.b64decode(BLOB), N, B)


def main():
    s = reconstruir()
    if '--bits' in sys.argv:
        sys.stdout.write(s)
        return
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
            .replace("@@MODELO@@", MODELO_SRC)
            .replace("@@DECODIFICADOR@@", _DECODIFICADOR_SRC))


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
    """Propone contextos según log2(N); no garantiza un contexto óptimo."""
    if n < 2:
        raise ValueError("Se requieren al menos dos bits")
    log_n = int(math.log2(n))
    b_min = max(4, log_n - 2)
    b_max = min(n - 1, log_n + 16)
    
    # Muestreo con paso 4 dentro de la cota logarítmica
    rango = list(range(b_min, b_max + 1, 4))
    if not rango:
        rango = [n - 1]
    return rango


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('entrada', help='Cadena de bits o archivo; de PDF se comprime su texto extraído')
    parser.add_argument('contexto', type=int, nargs='?')
    parser.add_argument('--modo', choices=('rapido', 'completo'), default='rapido',
                        help='rapido: 2 contextos y hasta 4 reglas candidatas; completo: hasta 32 candidatas')
    parser.add_argument('--trabajadores', type=int, default=None,
                        help='Cantidad máxima de procesos (por defecto hasta 4 en entradas grandes)')
    parser.add_argument('--salida', default=None, help='Archivo binario .hfc de salida')
    args = parser.parse_args()
    if args.trabajadores is not None and args.trabajadores < 1:
        parser.error('--trabajadores debe ser positivo')

    es_archivo = os.path.isfile(args.entrada)
    s = leer_bits(args.entrada)
    n = len(s)
    if n < 2 + BITS_PRED:
        sys.exit(f"Hacen falta al menos {2 + BITS_PRED} bits.")

    if args.contexto is not None:
        if not 1 <= args.contexto < n:
            parser.error('El contexto debe estar entre 1 y len(s) - 1')
        opciones = [args.contexto]
    else:
        opciones = obtener_rango_optimo(n)
        if args.modo == 'rapido':
            opciones = opciones[:2]

    inicio = time.perf_counter()
    print(f"Búsqueda heurística ({args.modo}): evaluando {len(opciones)} contextos {opciones}")
    print(" b | reglas exactas | semilla | Trie | relleno | carga aritm. | cabecera | TOTAL (bits) | seg")
    
    mejor = None
    
    # Crear procesos cuesta más que comprimir entradas pequeñas y multiplica
    # la memoria del modelo. El límite explícito permite ajustar ambas cosas.
    limite = args.trabajadores if args.trabajadores is not None else (4 if n >= 65536 else 1)
    trabajadores = min(len(opciones), limite, os.cpu_count() or 1)
    grupo = ProcessPoolExecutor(max_workers=trabajadores) if trabajadores > 1 else nullcontext()
    with grupo as executor:
        tareas = [(s, b, args.modo) for b in opciones]
        resultados = executor.map(_evaluar_worker, tareas) if executor else map(_evaluar_worker, tareas)
        for info in resultados:
            print(f"{info['b']:2d} | {info['reglas']:14d} | {info['semilla']:7d} | {info['trie']:4d} | "
                  f"{info['relleno']:7d} | {info['carga']:12d} | {info['cabecera']:8d} | "
                  f"{info['total']} | {info['dt']:.2f}")
            if mejor is None or info["total"] < mejor["total"]:
                mejor = info

    print(f"\nElegido: b = {mejor['b']} | total: {mejor['total']} bits "
          f"({mejor['total'] / n:.4f} del original de {n} bits)")
    print(f"Tiempo de compresión: {time.perf_counter() - inicio:.3f} s")
    print("MDL = cabecera + semilla + trie + relleno + residuo; "
          "cota de Kolmogórov: K(s) <= total + constante del decodificador.")

    if es_archivo or args.salida:
        salida = args.salida or 'comprimido.hfc'
        with open(salida, 'wb') as f:
            f.write(serializar_paquete(mejor))
        print(f"Paquete binario guardado en {salida}")
    if es_archivo:
        inicio_codigo = time.perf_counter()
        codigo = generar_decodificador(s, mejor)
        print(f"Tiempo de generación del decodificador: {time.perf_counter() - inicio_codigo:.6f} s")
        with open("decodificador.py", "w", encoding="utf-8") as f:
            f.write(codigo)
        print("\nDecodificador guardado en decodificador.py")
        print("Al ejecutarlo debe imprimir: VERIFICACION SHA-256: OK")


if __name__ == "__main__":
    main()
