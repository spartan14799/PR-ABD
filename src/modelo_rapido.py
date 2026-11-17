"""Implicaciones probabilísticas por bloques, filtro MDL y codificación sin pérdida.

Cada tabla transmite P(bit=0 | posiciones siguientes). Las posiciones pueden
omitirse: las reglas admiten excepciones, que codifica el flujo aritmético.
El filtro estima el costo completo y solo codifica su mejor candidato.
Formato MR01 independiente de hibrido.py; NumPy solo se necesita al comprimir.
"""
import argparse
import base64
import inspect
import math
from pathlib import Path
import time
import zlib

import numpy as np

MAGIC = b'MR\x01'


def _entero(n):
    salida = bytearray()
    while n >= 128:
        salida.append((n & 127) | 128)
        n >>= 7
    salida.append(n)
    return bytes(salida)


def _bits_bytes(s):
    return int(s.ljust((len(s) + 7) // 8 * 8, '0') or '0', 2).to_bytes((len(s) + 7) // 8, 'big')


def _bytes_bits(datos, n):
    return ''.join(format(v, '08b') for v in datos)[:n]


def _registro(modo, carga):
    return bytes([modo]) + _entero(len(carga)) + carga


def _codificar(eventos):
    """Codificador de rango: cada evento contiene bit y P(0) en escala 65536."""
    out = bytearray()
    low, rng, cache, size = 0, 0xFFFFFFFF, 0, 1

    def shift_low():
        nonlocal low, cache, size
        if low < 0xFF000000 or low > 0xFFFFFFFF:
            carry = low >> 32
            temp = cache
            while True:
                out.append((temp + carry) & 255)
                temp = 255
                size -= 1
                if not size:
                    break
            cache = (low >> 24) & 255
        size += 1
        low = (low & 0xFFFFFF) << 8

    for bit, p0 in eventos:
        bound = (rng >> 16) * p0
        if bit:
            low += bound
            rng -= bound
        else:
            rng = bound
        while rng < (1 << 24):
            rng <<= 8
            shift_low()
    hi = low + rng - 1
    for k in range(32, -1, -1):
        v = ((low + (1 << k) - 1) >> k) << k
        if v <= hi:
            low = v
            break
    for _ in range(5):
        shift_low()
    return bytes(out[1:]).rstrip(b'\x00')


def _decodificar(carga, n, b):
    if not carga:
        raise ValueError('Modelo truncado')
    k = carga[0]
    if k > min(b, 8):
        raise ValueError('Demasiadas posiciones')
    posiciones = list(carga[1:1+k])
    if len(posiciones) != k or posiciones != sorted(set(posiciones)) or any(j >= b for j in posiciones):
        raise ValueError('Posiciones inválidas')
    fin_tabla = 1 + k + (1 << k)
    fin_semilla = fin_tabla + (b + 7) // 8
    if len(carga) < fin_semilla:
        raise ValueError('Tabla o semilla truncadas')
    tabla = carga[1+k:fin_tabla]
    if 0 in tabla:
        raise ValueError('Probabilidad nula')
    semilla = _bytes_bits(carga[fin_tabla:fin_semilla], b)
    datos = carga[fin_semilla:]
    ip = 0

    def siguiente():
        nonlocal ip
        v = datos[ip] if ip < len(datos) else 0
        ip += 1
        return v

    code = 0
    for _ in range(4):
        code = (code << 8) | siguiente()
    rng = 0xFFFFFFFF
    contexto = int(semilla, 2)
    res = list(semilla[::-1])
    for _ in range(n-b):
        indice = 0
        for j in posiciones:
            indice = (indice << 1) | ((contexto >> (b-j-1)) & 1)
        bound = (rng >> 16) * (tabla[indice] << 8)
        if code < bound:
            bit = 0
            rng = bound
        else:
            bit = 1
            code -= bound
            rng -= bound
        while rng < (1 << 24):
            rng <<= 8
            code = (code << 8) | siguiente()
        res.append(str(bit))
        contexto = (bit << (b-1)) | (contexto >> 1)
    return ''.join(reversed(res))


def desempaquetar(paquete):
    """Lee el contenedor completo y verifica CRC32 (corrupción accidental)."""
    if not paquete.startswith(MAGIC):
        raise ValueError('Formato o versión incorrectos')
    pos = len(MAGIC)

    def entero():
        nonlocal pos
        valor = 0
        for desplazamiento in range(0, 70, 7):
            if pos >= len(paquete):
                raise ValueError('Cabecera truncada')
            byte = paquete[pos]
            pos += 1
            valor |= (byte & 127) << desplazamiento
            if byte < 128:
                if desplazamiento and byte == 0:
                    raise ValueError('Entero no canónico')
                return valor
        raise ValueError('Entero demasiado largo')

    n, b, bloque = entero(), entero(), entero()
    if n < 1 or not 1 <= b <= 64 or bloque < 1:
        raise ValueError('Parámetros inválidos')
    if len(paquete) < pos + 4:
        raise ValueError('CRC truncado')
    crc = int.from_bytes(paquete[pos:pos+4], 'big')
    pos += 4
    salida = []
    for inicio in range(0, n, bloque):
        cantidad = min(bloque, n-inicio)
        if pos >= len(paquete):
            raise ValueError('Bloque ausente')
        modo = paquete[pos]
        pos += 1
        tamano = entero()
        if pos + tamano > len(paquete):
            raise ValueError('Bloque truncado')
        carga = paquete[pos:pos+tamano]
        pos += tamano
        if modo == 0:
            if tamano != (cantidad+7)//8:
                raise ValueError('Longitud literal inválida')
            s = _bytes_bits(carga, cantidad)
        elif modo == 1 and cantidad > b:
            s = _decodificar(carga, cantidad, b)
        else:
            raise ValueError('Modo de bloque inválido')
        salida.append(s)
    if pos != len(paquete):
        raise ValueError('Bytes sobrantes')
    resultado = ''.join(salida)
    if zlib.crc32(resultado.encode('ascii')) != crc:
        raise ValueError('CRC incorrecto')
    return resultado


def filtrar_modelos(s, b, *, ancho=8, max_atributos=3):
    """Búsqueda acotada de tablas; no ejecuta el codificador aritmético.

    Estima -log2(P) usando exactamente las probabilidades cuantizadas enviadas,
    más tabla, posiciones, semilla, registro y cuatro bytes de margen de rango.
    Las cuentas incluyen el bloque: se transmiten, no se suponen conocidas por
    el decodificador. Su entrenamiento no implica predicción causal de cuentas.
    """
    arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8) & 1
    n = len(s)-b
    valores = arr[:n]
    cache = {}

    def evaluar(posiciones):
        if posiciones in cache:
            return cache[posiciones]
        ids = np.zeros(n, dtype=np.int64)
        for j in posiciones:
            ids = (ids << 1) | arr[j+1:j+1+n]
        total = np.bincount(ids, minlength=1 << len(posiciones))
        unos = np.bincount(ids, weights=valores, minlength=len(total))
        ceros = total-unos
        tabla = np.clip(np.rint((ceros + .5)*256/(total+1)), 1, 255).astype(np.uint8)
        p0 = tabla.astype(float)/256
        entropia = float(-(ceros*np.log2(p0) + unos*np.log2(1-p0)).sum())
        metadata = 1 + len(posiciones) + len(tabla) + (b+7)//8
        bytes_estimados = metadata + math.ceil(entropia/8) + 4
        costo = 8*(1 + len(_entero(bytes_estimados)) + bytes_estimados)
        modelo = dict(posiciones=posiciones, tabla=tabla.tobytes(), estimado=costo)
        cache[posiciones] = modelo
        return modelo

    evaluar(())  # Frecuencia marginal, útil cuando hay sesgo sin contexto.
    frontera = [()]
    for _ in range(min(max_atributos, b)):
        siguientes = set()
        for posiciones in frontera:
            for j in range(b):
                if j not in posiciones:
                    siguientes.add(tuple(sorted((*posiciones, j))))
        frontera = sorted(siguientes, key=lambda p: (evaluar(p)['estimado'], p))[:ancho]
    # Complementa posiciones dispersas con prefijos que distinguen fases largas.
    for k in range(1, min(b, 8)+1):
        evaluar(tuple(range(k)))
    mejor = min(cache.values(), key=lambda m: (m['estimado'], m['posiciones']))
    return mejor, len(cache)


def _comprimir_bloque(s, b):
    literal = _registro(0, _bits_bytes(s))
    estadistica = dict(n=len(s), modo='literal', candidatos=0, codificados=0,
                       posiciones=[], estimado=None, total=8*len(literal))
    if len(s) <= b:
        return literal, estadistica
    modelo, cantidad = filtrar_modelos(s, b)
    estadistica.update(candidatos=cantidad, estimado=modelo['estimado'])
    if modelo['estimado'] >= 8*len(literal):
        return literal, estadistica
    posiciones, tabla = modelo['posiciones'], modelo['tabla']
    arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8) & 1
    n = len(s)-b
    ids = np.zeros(n, dtype=np.int64)
    for j in posiciones:
        ids = (ids << 1) | arr[j+1:j+1+n]
    probs = np.frombuffer(tabla, dtype=np.uint8).astype(np.int64)[ids] << 8
    carga = _codificar(zip(arr[:n][::-1].tolist(), probs[::-1].tolist()))
    metadata = bytes([len(posiciones), *posiciones]) + tabla + _bits_bytes(s[-b:])
    comprimido = _registro(1, metadata+carga)
    estadistica['codificados'] = 1
    if len(comprimido) < len(literal):
        estadistica.update(modo='probabilistico', posiciones=list(posiciones),
                           total=8*len(comprimido))
        return comprimido, estadistica
    return literal, estadistica


def empaquetar(s, b=16, *, bloque=8192):
    """Comprime bits; total incluye cabecera, CRC, tablas y bloques literales."""
    if not s or s.strip('01'):
        raise ValueError('Se requiere una cadena no vacía de ceros y unos')
    if not 1 <= b <= 64 or bloque < 1:
        raise ValueError('Contexto entre 1 y 64; bloque positivo')
    cabecera = MAGIC + _entero(len(s)) + _entero(b) + _entero(bloque)
    cabecera += zlib.crc32(s.encode('ascii')).to_bytes(4, 'big')
    registros, estadisticas = [], []
    for inicio in range(0, len(s), bloque):
        registro, estado = _comprimir_bloque(s[inicio:inicio+bloque], b)
        registros.append(registro)
        estadisticas.append(estado)
    paquete = cabecera + b''.join(registros)
    return dict(n=len(s), b=b, bloque=bloque, blob=paquete, total=8*len(paquete),
                cabecera=8*len(cabecera), bloques=estadisticas)


def serializar_paquete(info):
    return info['blob']


def generar_decodificador(s, info):
    """Programa autónomo sin NumPy ni archivos auxiliares; --bits emite los bits."""
    if info['n'] != len(s):
        raise ValueError('Longitud incompatible')
    import hashlib
    funciones = (_bytes_bits, _decodificar, desempaquetar)
    fuente = '\n\n'.join(inspect.getsource(f) for f in funciones)
    return ("import base64, hashlib, sys, zlib\nMAGIC = b'MR\\x01'\n" + fuente +
            '\nPAQUETE = ' + repr(base64.b64encode(info['blob'])) +
            '\ndef reconstruir():\n    return desempaquetar(base64.b64decode(PAQUETE))\n' +
            "\nif __name__ == '__main__':\n    s = reconstruir()\n" +
            f"    assert hashlib.sha256(s.encode()).hexdigest() == {hashlib.sha256(s.encode()).hexdigest()!r}\n" +
            "    if '--bits' in sys.argv:\n        sys.stdout.write(s)\n" +
            "    else:\n        print('VERIFICACION SHA-256: OK')\n")


def leer_bits(entrada):
    ruta = Path(entrada)
    # Evita consultar el filesystem con una cadena de bits arbitrariamente larga.
    if entrada and not entrada.strip('01'):
        return entrada
    if ruta.suffix.lower() == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        datos = ''.join(p.extract_text() or '' for p in PdfReader(ruta).pages).encode('utf-8')
    else:
        datos = ruta.read_bytes()
    return ''.join(format(v, '08b') for v in datos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('entrada', help='Bits o archivo; de un PDF se comprime el texto extraído')
    parser.add_argument('contexto', nargs='?', type=int, default=16)
    parser.add_argument('--bloque', type=int, default=8192, help='Bits por bloque')
    parser.add_argument('--salida', default='comprimido.mrp')
    parser.add_argument('--decodificador', help='Ruta opcional del Python autónomo')
    parser.add_argument('--descomprimir', action='store_true', help='Emite bits desde un paquete MR01')
    args = parser.parse_args()
    if args.descomprimir:
        print(desempaquetar(Path(args.entrada).read_bytes()), end='')
        return
    try:
        s = leer_bits(args.entrada)
        inicio = time.perf_counter()
        info = empaquetar(s, args.contexto, bloque=args.bloque)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    duracion = time.perf_counter()-inicio
    Path(args.salida).write_bytes(serializar_paquete(info))
    if args.decodificador:
        Path(args.decodificador).write_text(generar_decodificador(s, info))
    print(f"Original: {len(s)} bits | Paquete: {info['total']} bits | Tiempo: {duracion:.4f} s")
    ahorro = 100 * (len(s) - info['total']) / len(s)
    if ahorro >= 0:
        print(f"Ahorro respecto al original: {ahorro:.2f}% (incluye cabecera y metadatos)")
    else:
        print(f"Aumento respecto al original: {-ahorro:.2f}% (incluye cabecera y metadatos)")
    print(f"Bloques probabilísticos: {sum(x['modo'] == 'probabilistico' for x in info['bloques'])}/{len(info['bloques'])}")
    print(f"Candidatos: {sum(x['candidatos'] for x in info['bloques'])} | Codificados: {sum(x['codificados'] for x in info['bloques'])}")


if __name__ == '__main__':
    main()
