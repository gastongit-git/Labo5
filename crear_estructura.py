from pathlib import Path

raiz = Path(__file__).resolve().parent

subcarpetas = [
    "datos/crudos",
    "datos/procesados",
    "adquisicion",
    "analisis/G",
    "analisis/I",
    "analisis/V",
    "analisis/compartido",
    "resultados/G",
    "resultados/I",
    "resultados/V",
    "resultados/finales",
]

for numero in range(1, 5):
    experimento = raiz / f"{numero:02d}_experimento"

    for subcarpeta in subcarpetas:
        carpeta = experimento / subcarpeta
        carpeta.mkdir(parents=True, exist_ok=True)

        # Permite que Git registre carpetas vacías
        (carpeta / ".gitkeep").touch(exist_ok=True)

herramientas = raiz / "herramientas_comunes"
herramientas.mkdir(exist_ok=True)
(herramientas / ".gitkeep").touch(exist_ok=True)

print("Estructura creada correctamente.")
