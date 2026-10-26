import subprocess
import sys
import random

def texto_a_bits(texto: str) -> str:
    return ''.join(format(b, '08b') for b in texto.encode('utf-8'))

def generar_thue_morse(n_bits: int) -> str:
    """Genera secuencia autosimilar/fractal sin periodicidad simple."""
    s = "0"
    while len(s) < n_bits:
        s += "".join("1" if c == "0" else "0" for c in s)
    return s[:n_bits]

CASOS_DE_PRUEBA = {
    "Patrón Periódico Simple (0101...)": "01" * 1000,
    "Patrón Periódico Lógico (11001100...)": "11001100" * 250,
    "Cadenas Largas (Bloques 0s y 1s)": ("0" * 500 + "1" * 500) * 2,
    "Datos Dispersos (95% 0s, 5% 1s)": ("0" * 95 + "1" * 5) * 20,
    "Texto Real ASCII (Repetitivo)": texto_a_bits("a" * 200),
    "Texto Real Español (Quijote)": texto_a_bits("En un lugar de la Mancha, de cuyo nombre no quiero acordarme... " * 10),
    "Código Fuente Python": texto_a_bits("def suma(a, b):\n    return a + b\n" * 20),
    "Secuencia Thue-Morse (Fractal)": generar_thue_morse(2000),
    "Aleatorio Puro (Entropía Máxima)": "".join(random.choices(["0", "1"], k=2000)),
}


def probar_algoritmo(script_nombre="modelo_rapido.py"):
    print(f"=== EJECUTANDO BATERÍA DE PRUEBAS SOBRE: {script_nombre} ===\n")
    
    for nombre, bits in CASOS_DE_PRUEBA.items():
        print(f"--> Prueba: {nombre}")
        print(f"    Longitud original: {len(bits)} bits")
        try:
            subprocess.run([sys.executable, script_nombre, bits], check=True)
        except FileNotFoundError:
            print(f"Error: No se encontró '{script_nombre}'. Verifique la ruta.")
            break
        except subprocess.CalledProcessError as e:
            print(f"Error durante la ejecución del caso: {e}")
        print("-" * 60)

if __name__ == "__main__":
    probar_algoritmo("modelo_rapido.py")
