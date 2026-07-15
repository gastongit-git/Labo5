from pathlib import Path
from datetime import datetime
import subprocess


def _buscar_repositorio(archivo):
    for carpeta in archivo.parents:
        if (carpeta / ".git").exists():
            return carpeta

    raise ValueError("El archivo no está dentro de un repositorio Git.")


def subir_a_github(archivo, mensaje=None):
    archivo = Path(archivo).resolve()

    if not archivo.exists():
        print("El archivo no existe:", archivo)
        return False

    try:
        repo = _buscar_repositorio(archivo)
        ruta_relativa = archivo.relative_to(repo)

        if mensaje is None:
            ahora = datetime.now()
            mensaje = f"Medición automática {ahora:%Y-%m-%d %H:%M:%S}"

        subprocess.run(
            ["git", "add", str(ruta_relativa)],
            cwd=repo,
            check=True
        )

        cambios = subprocess.run(
            [
                "git",
                "diff",
                "--cached",
                "--quiet",
                "--",
                str(ruta_relativa)
            ],
            cwd=repo
        )

        if cambios.returncode == 0:
            print("El archivo no tiene cambios nuevos.")
            return False

        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                mensaje,
                "--",
                str(ruta_relativa)
            ],
            cwd=repo,
            check=True
        )

        subprocess.run(
            ["git", "pull", "--rebase", "--autostash"],
            cwd=repo,
            check=True
        )

        subprocess.run(
            ["git", "push"],
            cwd=repo,
            check=True
        )

        print("Archivo subido correctamente:", ruta_relativa)
        return True

    except (subprocess.CalledProcessError, ValueError) as error:
        print("El archivo quedó guardado localmente, pero falló la subida.")
        print(error)
        return False