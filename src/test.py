import contextlib
import io
import sys
from math import ceil, log2


def reglas_utiles(s, b):
    n = len(s)
    cuentas = {}
    for i in range(n - b):
        prev = s[i] == "1"
        for m in range(1, b + 1):
            w = s[i + 1:i + 1 + m]
            c = cuentas.get(w)
            if c is None:
                c = cuentas[w] = [0, 0]
            c[prev] += 1
    deterministicas = set()
    reglas = {}
    for w in sorted(cuentas, key=lambda x: (len(x), x)):
        c = cuentas[w]
        if c[0] and c[1]:
            continue
        if any(w[:k] in deterministicas for k in range(1, len(w))):
            continue
        deterministicas.add(w)
        if c[0] + c[1] > costo_regla(w, b):
            reglas[w] = "1" if c[1] else "0"
    return dict(sorted(reglas.items(), key=lambda kv: (len(kv[0]), kv[0])))


def costo_regla(w, b):
    return len(w) + 1 + ceil(log2(b + 1))


def bit_deducido(contexto, reglas, b):
    for m in range(1, b + 1):
        v = reglas.get(contexto[:m])
        if v is not None:
            return v
    return None


def codificar(s, b, reglas):
    n = len(s)
    datos = []
    for i in range(n - b - 1, -1, -1):
        v = bit_deducido(s[i + 1:i + 1 + b], reglas, b)
        if v is None:
            datos.append(s[i])
    return s[n - b:], "".join(datos)


def evaluar(s, b):
    reglas = reglas_utiles(s, b)
    semilla, datos = codificar(s, b, reglas)
    costo = sum(costo_regla(w, b) for w in reglas)
    return reglas, semilla, datos, costo, len(semilla) + len(datos) + costo


def generar_codigo(n, b, reglas, semilla, datos):
    lineas = [
        f"n = {n}",
        f"b = {b}",
        f"reglas = {reglas!r}",
        f"semilla = {semilla!r}",
        f"datos = {datos!r}",
        "r = list(semilla[::-1])",
        "pos = 0",
        "while len(r) < n:",
        "    c = ''.join(r[-1:-b - 1:-1])",
        "    bit = None",
        "    for m in range(1, b + 1):",
        "        if c[:m] in reglas:",
        "            bit = reglas[c[:m]]",
        "            break",
        "    if bit is None:",
        "        bit = datos[pos]",
        "        pos += 1",
        "    r.append(bit)",
        "cadena = ''.join(r[::-1])",
        "print(cadena)",
    ]
    return "\n".join(lineas)


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("Uso: python deslizante_bits.py BITS [BITS_CONTEXTO]")
    s = "".join(c for c in sys.argv[1] if c in "01")
    n = len(s)
    if n < 2:
        sys.exit("Hacen falta al menos 2 bits.")

    if len(sys.argv) == 3:
        opciones = [int(sys.argv[2])]
        if not 1 <= opciones[0] < n:
            sys.exit("BITS_CONTEXTO debe estar entre 1 y n-1.")
    else:
        opciones = list(range(1, min(16, n - 1) + 1))

    mejor = None
    if len(opciones) > 1:
        print(" b | reglas | datos | semilla | costo reglas | total")
    for b in opciones:
        res = evaluar(s, b)
        reglas, semilla, datos, costo, total = res
        if len(opciones) > 1:
            print(f"{b:2d} | {len(reglas):6d} | {len(datos):5d} | {len(semilla):7d} | "
                  f"{costo:12d} | {total}")
        if mejor is None or total < mejor[1][4]:
            mejor = (b, res)

    b, (reglas, semilla, datos, costo, total) = mejor
    print(f"\nContexto elegido: b = {b}")
    print("=== REGLAS: si los bits siguientes empiezan por W, el bit anterior es V ===")
    if not reglas:
        print("(ninguna regla compensa su costo)")
    for w, v in reglas.items():
        print(f"siguientes {w}  →  anterior {v}")

    codigo = generar_codigo(n, b, reglas, semilla, datos)
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(codigo, ns)
    ok = ns.get("cadena") == s

    print("\n=== RESUMEN ===")
    print(f"Original: {n} bits | semilla: {len(semilla)} | datos: {len(datos)} | "
          f"reglas: {costo} ({len(reglas)} reglas) | total: {total}")
    print(f"Bits deducidos: {n - b - len(datos)} de {n - b}")
    print(f"Verificación: {'la cadena se regenera exacta' if ok else 'FALLA, la cadena no coincide'}")

    print("\n=== CÓDIGO QUE REGENERA LA CADENA ===")
    print(codigo)


if __name__ == "__main__":
    main()
