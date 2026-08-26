
import sys
from pathlib import Path

import cv2
import numpy as np

from scipy.ndimage import generic_filter
from openpiv import windef, filters


# ============================================================
# CONFIGURACIÓN
# ============================================================

VIDEO = r"C:\Users\user\Desktop\Labo5\1_Fluidos\datos\crudos\video3_brillantina_oscura.MOV"

# ROI
x0 = 204
y0 = 562
w = 817
h = 808

# Máscara circular
cx = 397
cy = 390
radio = 427.69147758635546

# Calibración
distancia_real_mm = 147
distancia_px = 1514

mm_por_px = distancia_real_mm / distancia_px

# Validación
n_std = 8
umbral_mediana = 6
epsilon = 0.1


# ============================================================
# MÁSCARA
# ============================================================

Y, X = np.ogrid[:h, :w]

mask = (X - cx)**2 + (Y - cy)**2 > radio**2


# ============================================================
# OPENPIV
# ============================================================

settings = windef.PIVSettings()

settings.windowsizes = (96, 64)
settings.overlap = (48, 32)
settings.num_iterations = 2

settings.correlation_method = "circular"
settings.subpixel_method = "gaussian"
settings.deformation_method = "symmetric"

settings.static_mask = mask

settings.sig2noise_validate = False

settings.min_max_u_disp = (-100, 100)
settings.min_max_v_disp = (-100, 100)

settings.std_threshold = 20
settings.median_threshold = 10


# ============================================================
# FUNCIONES
# ============================================================

def filtro_mediana_nan(a, size=3):

    return generic_filter(
        a,
        np.nanmedian,
        size=size,
        mode="nearest"
    )


def procesar_par(frame1, frame2, fps):

    # --------------------------------------------------------
    print("  A - gris y ROI", flush=True)
    # --------------------------------------------------------

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    f1 = frame1[y0:y0+h, x0:x0+w].astype(np.int32)
    f2 = frame2[y0:y0+h, x0:x0+w].astype(np.int32)


    # --------------------------------------------------------
    print("  B - pass 1", flush=True)
    # --------------------------------------------------------

    x1, y1, du1, dv1, s2n1 = windef.first_pass(
        f1,
        f2,
        settings
    )


    # --------------------------------------------------------
    print("  C - mascara pass 1", flush=True)
    # --------------------------------------------------------

    xi1 = np.rint(x1).astype(int)
    yi1 = np.rint(y1).astype(int)

    xi1 = np.clip(xi1, 0, mask.shape[1] - 1)
    yi1 = np.clip(yi1, 0, mask.shape[0] - 1)

    mask1 = mask[yi1, xi1]

    du1 = np.ma.array(du1, mask=mask1)
    dv1 = np.ma.array(dv1, mask=mask1)


    # --------------------------------------------------------
    print("  D - pass 2", flush=True)
    # --------------------------------------------------------

    x2, y2, du2, dv2, mask2, flags2 = windef.multipass_img_deform(
        f1,
        f2,
        1,
        x1,
        y1,
        du1,
        dv1,
        settings
    )


    # --------------------------------------------------------
    print("  E - validacion", flush=True)
    # --------------------------------------------------------

    du = np.asarray(
        np.ma.filled(du2, np.nan),
        dtype=float
    )

    dv = np.asarray(
        np.ma.filled(dv2, np.nan),
        dtype=float
    )

    mask_geom = np.asarray(
        mask2,
        dtype=bool
    )

    validos = (
        ~mask_geom
        & np.isfinite(du)
        & np.isfinite(dv)
    )


    # =========================
    # STD
    # =========================

    valores_u = np.where(
        validos,
        du,
        np.nan
    )

    valores_v = np.where(
        validos,
        dv,
        np.nan
    )

    mean_u = np.nanmean(valores_u)
    mean_v = np.nanmean(valores_v)

    std_u = np.nanstd(valores_u)
    std_v = np.nanstd(valores_v)

    bad_std = (
        (du < mean_u - n_std * std_u)
        | (du > mean_u + n_std * std_u)
        | (dv < mean_v - n_std * std_v)
        | (dv > mean_v + n_std * std_v)
    )


    # =========================
    # MEDIANA LOCAL
    # =========================

    med_u = filtro_mediana_nan(
        valores_u,
        size=3
    )

    med_v = filtro_mediana_nan(
        valores_v,
        size=3
    )

    res_u = np.abs(du - med_u)
    res_v = np.abs(dv - med_v)

    med_res_u = filtro_mediana_nan(
        np.where(
            validos,
            res_u,
            np.nan
        ),
        size=3
    )

    med_res_v = filtro_mediana_nan(
        np.where(
            validos,
            res_v,
            np.nan
        ),
        size=3
    )

    norm_u = res_u / (med_res_u + epsilon)
    norm_v = res_v / (med_res_v + epsilon)

    residuo = np.sqrt(
        norm_u**2 + norm_v**2
    )

    bad_mediana = (
        residuo > umbral_mediana
    )


    # =========================
    # RECHAZOS
    # =========================

    rechazados = (
        validos
        & (bad_std | bad_mediana)
    )

    n_validos = np.sum(validos)

    if n_validos > 0:

        porcentaje = (
            100
            * np.sum(rechazados)
            / n_validos
        )

    else:

        porcentaje = np.nan


    # --------------------------------------------------------
    print("  F - interpolacion outliers", flush=True)
    # --------------------------------------------------------

    du_interp, dv_interp = filters.replace_outliers(
        du2,
        dv2,
        flags=rechazados,
        method="localmean",
        max_iter=3,
        kernel_size=1
    )

    du_interp = np.ma.array(
        du_interp,
        mask=mask2
    )

    dv_interp = np.ma.array(
        dv_interp,
        mask=mask2
    )


    # =========================
    # CALIBRACIÓN A mm/s
    # =========================

    u = (
        du_interp
        * mm_por_px
        * fps
    )

    v = (
        dv_interp
        * mm_por_px
        * fps
    )


    # --------------------------------------------------------
    print("  G - terminado", flush=True)
    # --------------------------------------------------------

    return (
        x2,
        y2,
        np.ma.filled(u, np.nan),
        np.ma.filled(v, np.nan),
        rechazados,
        porcentaje
    )


# ============================================================
# PROCESAR UN BLOQUE
# ============================================================

def procesar_bloque(inicio, fin, salida):

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir el video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    # Nos ubicamos en el primer frame del bloque
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        inicio
    )

    ok, frame_anterior = cap.read()

    if not ok:
        raise RuntimeError(
            f"No se pudo leer el frame {inicio}"
        )


    campos_u = []
    campos_v = []

    campos_rechazados = []
    porcentajes = []

    x_grid = None
    y_grid = None


    for par in range(inicio, fin):

        print(
            f"Par {par} -> {par + 1}",
            flush=True
        )

        ok, frame_actual = cap.read()

        if not ok:
            raise RuntimeError(
                f"No se pudo leer el frame {par + 1}"
            )

        (
            x,
            y,
            u,
            v,
            rechazados,
            porcentaje
        ) = procesar_par(
            frame_anterior,
            frame_actual,
            fps
        )


        if x_grid is None:

            x_grid = x
            y_grid = y


        campos_u.append(u)
        campos_v.append(v)

        campos_rechazados.append(
            rechazados
        )

        porcentajes.append(
            porcentaje
        )


        frame_anterior = frame_actual


    cap.release()


    # ========================================================
    # GUARDAR
    # ========================================================

    np.savez_compressed(
        salida,

        x=x_grid,
        y=y_grid,

        u=np.stack(campos_u),
        v=np.stack(campos_v),

        rechazados=np.stack(
            campos_rechazados
        ),

        porcentaje_rechazado=np.array(
            porcentajes
        ),

        fps=fps,
        mm_por_px=mm_por_px,

        par_inicial=inicio,
        par_final=fin
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    inicio = int(sys.argv[1])
    fin = int(sys.argv[2])
    salida = Path(sys.argv[3])

    procesar_bloque(
        inicio,
        fin,
        salida
    )

    print(
        f"OK: pares {inicio} a {fin - 1}",
        flush=True
    )
