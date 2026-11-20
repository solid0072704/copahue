"""Armado del video: animación cuadro a cuadro, transiciones entre láminas y barrido de las portadas."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from curso_youtube import ensamblar

FPS = ensamblar.FPS
TAM = (64, 36)


def _imagen(ruta: Path, color) -> None:
    Image.new("RGB", TAM, color).save(ruta, quality=95) if ruta.suffix == ".jpg" else Image.new("RGB", TAM, color).save(ruta)


def _trabajo(tmp_path: Path, animada: bool) -> tuple[Path, dict]:
    """Dos láminas: una de contenido con dos pasos y una portada de capítulo. Devuelve la carpeta y la línea de tiempo."""
    frames, html = tmp_path / "frames", tmp_path / "html"
    frames.mkdir()
    html.mkdir()
    (html / "laminas.json").write_text(json.dumps([
        dict(id="m00_s01", tipo="puntos", pasos=2, continua=False),
        dict(id="m01_s01", tipo="seccion", pasos=1, continua=True)]))
    pasos = {"m00_s01_p00": 6, "m00_s01_p01": 5, "m01_s01_p00": 40}
    for clave, n in pasos.items():
        _imagen(frames / f"{clave}.png", (20, 30, 50))
        if animada:
            _imagen(frames / f"{clave}.jpg", (20, 30, 50))
            for f in range(n):
                _imagen(frames / f"{clave}_f{f:03d}.jpg", (20 + f, 30, 50))
    if animada:
        (frames / "m00_s01.listo").write_text(json.dumps(dict(firma="x", cuadros={k: v for k, v in pasos.items() if k.startswith("m00")})))
        (frames / "m01_s01.listo").write_text(json.dumps(dict(firma="x", cuadros={"m01_s01_p00": 40})))
    tl = dict(total=6.0, eventos=[dict(t=0.35, frame="m00_s01_p00.png"), dict(t=2.0, frame="m00_s01_p01.png"),
                                  dict(t=4.0, frame="m01_s01_p00.png")])
    return tmp_path, tl


def test_la_secuencia_cubre_el_video_completo_con_su_animacion(tmp_path):
    trabajo, tl = _trabajo(tmp_path, animada=True)
    seq = ensamblar.secuencia(tl, trabajo)
    assert sum(n for _, n, _ in seq) == round(tl["total"] * FPS)
    assert all(img.exists() and img.suffix == ".jpg" for img, _, _ in seq)      # un solo formato de imagen
    nombres = [img.name for img, _, _ in seq]
    # primer paso: sus seis cuadros y después la imagen final hasta el segundo 2
    assert nombres[:7] == [f"m00_s01_p00_f{f:03d}.jpg" for f in range(6)] + ["m00_s01_p00.jpg"]
    assert seq[6][1] == 60 - 6
    # el segundo paso de la misma lámina entra sin fundido: su animación parte de la imagen anterior
    assert nombres[7:13] == [f"m00_s01_p01_f{f:03d}.jpg" for f in range(5)] + ["m00_s01_p01.jpg"]
    # la portada del capítulo entra con barrido sobre sus primeros cuadros y después sigue su animación
    portada = [x for x in seq if x[2] == "m01_s01"]
    assert all("_desde_m00_s01_p01_t" in img.name for img, _, _ in portada[:ensamblar.BARRIDO])
    assert portada[ensamblar.BARRIDO][0].name == f"m01_s01_p00_f{ensamblar.BARRIDO:03d}.jpg"
    assert portada[-1][0].name == "m01_s01_p00.jpg"


def test_las_transiciones_no_se_reescriben_si_nada_cambio(tmp_path):
    trabajo, tl = _trabajo(tmp_path, animada=True)
    primera = {img: img.stat().st_mtime_ns for img, _, _ in ensamblar.secuencia(tl, trabajo)}
    segunda = {img: img.stat().st_mtime_ns for img, _, _ in ensamblar.secuencia(tl, trabajo)}
    assert primera == segunda          # así un tramo ya codificado conserva su firma y no se repite


def test_sin_animacion_se_arma_con_las_imagenes_fijas(tmp_path):
    trabajo, tl = _trabajo(tmp_path, animada=False)
    seq = ensamblar.secuencia(tl, trabajo)
    assert sum(n for _, n, _ in seq) == round(tl["total"] * FPS)
    assert all(img.suffix == ".jpg" and img.parent.name == "fundidos" for img, _, _ in seq)
    assert seq[0][0].name == "m00_s01_p00.jpg" and seq[0][1] == 60
    # entre imágenes fijas hay un fundido corto; la portada entra igual con barrido
    assert [n for _, n, _ in seq[1:5]] == [1, 1, 1, 1]
    assert sum(1 for img, _, _ in seq if "_desde_m00_s01_p01_t" in img.name) == ensamblar.BARRIDO


def test_el_barrido_descubre_de_izquierda_a_derecha():
    a, b = Image.new("RGB", (400, 100), (200, 0, 0)), Image.new("RGB", (400, 100), (0, 0, 200))
    mitad = ensamblar._barrer(a, b, 0.5)
    assert mitad.getpixel((50, 50)) == (0, 0, 200)                 # a la izquierda ya está la lámina nueva
    assert mitad.getpixel((199, 50)) == ensamblar.ORO              # el filete marca el borde
    assert mitad.getpixel((399, 50)) == (200, 0, 0)                # a la derecha sigue la anterior
    sombra = mitad.getpixel((205, 50))
    assert sombra[0] < 200 and sombra[2] <= 14                     # oscurecida junto al borde
    assert ensamblar._barrer(a, b, 0.0).getpixel((300, 50)) == (200, 0, 0)
    assert ensamblar._barrer(a, b, 1.0).getpixel((10, 50)) == (0, 0, 200)


def test_el_motor_de_movimiento_y_el_capturador_son_javascript_valido():
    if not shutil.which("node"):
        pytest.skip("requiere Node")
    aqui = Path(ensamblar.__file__).parent
    for archivo in ("movimiento.js", "capturar.cjs"):
        r = subprocess.run(["node", "--check", str(aqui / archivo)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_cada_imagen_cae_en_su_cuadro_exacto(tmp_path):
    """Ni se pierde ni se repite un cuadro: la animación a 30 cuadros por segundo se codifica uno a uno."""
    if not shutil.which("ffmpeg"):
        pytest.skip("requiere ffmpeg")
    niveles = [30 + 12 * k for k in range(18)]
    duraciones = [1] * 13 + [9] + [1] * 3 + [5]          # animación, una pausa, más animación y la imagen final
    lista = []
    for k, (nivel, n) in enumerate(zip(niveles, duraciones)):
        ruta = tmp_path / f"c{k:02d}.jpg"
        Image.new("RGB", TAM, (nivel, nivel, nivel)).save(ruta, quality=95)
        lista.append((ruta, n))
    video = tmp_path / "tramo.mp4"
    ensamblar._codificar(lista, video)
    # se lee la luminancia tal como quedó guardada (rango de video: 16 a 235), sin convertirla
    crudo = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=16:8:in_range=tv:out_range=tv,format=yuv420p",
                            "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    cuadro = 16 * 8 * 3 // 2
    vistos = [sum(crudo[i:i + 128]) / 128 for i in range(0, len(crudo), cuadro)]
    esperados = [16 + nivel * 219 / 255 for nivel, n in zip(niveles, duraciones) for _ in range(n)]
    assert len(vistos) == len(esperados) == sum(duraciones)
    assert all(abs(v - e) < 3 for v, e in zip(vistos, esperados)), [round(v - e, 1) for v, e in zip(vistos, esperados)]
