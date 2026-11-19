"""Une las láminas con la narración y produce el video (MP4 1080p), los capítulos y los subtítulos.

Uso:
    python -m curso_youtube.ensamblar <trabajo> <carpeta_audio> [--solo seg_000 seg_001 ...]
                                      [--salida video.mp4] [--borrador] [--fin 20]

- Espera un archivo de audio por segmento: <carpeta_audio>/seg_XXX.mp3 (o .wav). Los nombres y el texto de
  cada segmento salen de `python -m curso_youtube.guion --voz`.
- Las imágenes deben existir en <trabajo>/frames (se generan con `python -m curso_youtube.construir`).
- `--borrador` reemplaza los audios que faltan por silencio de la duración estimada, para revisar el ritmo.
- `--fin` agrega segundos de la última lámina al final (espacio para la pantalla final de YouTube).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from . import deck
from .voz import CARACTERES_POR_SEGUNDO

FPS = 30
MUESTRAS = 48000
VOLUMEN_OBJETIVO = -16.0   # LUFS
ENTRADA = 0.35             # segundos antes de la primera palabra
FUNDIDO = 4                # cuadros de fundido entre una imagen y la siguiente


def _ffmpeg(*args: str, capturar: bool = False) -> str:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    return r.stderr if capturar else ""


# --- Audio ----------------------------------------------------------------
def preparar_audio(origen: Path, destino: Path) -> float:
    """Convierte a WAV mono 48 kHz, recorta silencios de los extremos y nivela el volumen. Devuelve la duración."""
    medida = _ffmpeg("-i", str(origen), "-af", "ebur128", "-f", "null", "-", capturar=True)
    integrados = re.findall(r"I:\s+(-?\d+(?:\.\d+)?) LUFS", medida)
    ganancia = VOLUMEN_OBJETIVO - float(integrados[-1]) if integrados else 0.0
    recorte = ("silenceremove=start_periods=1:start_threshold=-48dB:start_silence=0.06,areverse,"
               "silenceremove=start_periods=1:start_threshold=-48dB:start_silence=0.12,areverse")
    _ffmpeg("-i", str(origen), "-af", f"{recorte},volume={ganancia:.2f}dB,alimiter=limit=0.84:level=false",
            "-ar", str(MUESTRAS), "-ac", "1", "-c:a", "pcm_s16le", str(destino))
    with wave.open(str(destino)) as w:
        return w.getnframes() / w.getframerate()


def silencios(wav: Path) -> list[tuple[float, float]]:
    salida = _ffmpeg("-i", str(wav), "-af", "silencedetect=noise=-38dB:d=0.10", "-f", "null", "-", capturar=True)
    inicios = [float(x) for x in re.findall(r"silence_start: (-?\d+(?:\.\d+)?)", salida)]
    fines = [float(x) for x in re.findall(r"silence_end: (-?\d+(?:\.\d+)?)", salida)]
    return [(a, b) for a, b in zip(inicios, fines) if b > a]


def alinear(pesos: list[float], nuevo_parrafo: list[bool], duracion: float, sil: list[tuple[float, float]]) -> list[float]:
    """Instante en que empieza cada paso dentro de un audio.

    Reparte la duración según el largo de cada paso y ajusta cada cambio a una pausa real de la voz
    (programación dinámica: minimiza la diferencia con la duración esperada y premia las pausas largas).
    """
    n = len(pesos)
    if n == 1:
        return [0.0]
    esperado = np.array(pesos, dtype=float) / sum(pesos) * duracion
    # Candidatos: fin de cada pausa (justo antes de que la voz retome) y una rejilla sin pausa como respaldo.
    cand = [(max(0.0, b - 0.06), min(b - a, 1.2)) for a, b in sil if 0.15 < b < duracion - 0.15]
    cand += [(t, 0.0) for t in np.arange(0.4, duracion - 0.2, 0.25)]
    cand.sort()
    tiempos = np.array([c[0] for c in cand])
    calidad = np.array([c[1] for c in cand])
    m = len(cand)
    INF = 1e18
    mejor = np.full((n, m), INF)
    previo = np.zeros((n, m), dtype=int)
    for i in range(1, n):
        premio = (2.4 if nuevo_parrafo[i] else 1.0) * np.minimum(calidad, 1.0 if nuevo_parrafo[i] else 0.5)
        castigo_sin_pausa = np.where(calidad == 0.0, 1.5 if nuevo_parrafo[i] else 0.4, 0.0)
        if i == 1:
            dur = tiempos
            costo = (dur - esperado[0]) ** 2 / (esperado[0] + 1.0)
            mejor[1] = np.where(dur > 0.2, costo, INF) - premio + castigo_sin_pausa
        else:
            dur = tiempos[None, :] - tiempos[:, None]            # [k_previo, k]
            costo = (dur - esperado[i - 1]) ** 2 / (esperado[i - 1] + 1.0)
            total = np.where(dur > 0.2, mejor[i - 1][:, None] + costo, INF)
            previo[i] = np.argmin(total, axis=0)
            mejor[i] = total[previo[i], np.arange(m)] - premio + castigo_sin_pausa
    final = mejor[n - 1] + (duracion - tiempos - esperado[n - 1]) ** 2 / (esperado[n - 1] + 1.0)
    k = int(np.argmin(final))
    cortes = []
    for i in range(n - 1, 0, -1):
        cortes.append(float(tiempos[k]))
        k = int(previo[i][k])
    return [0.0] + cortes[::-1]


def silencio_wav(destino: Path, segundos: float) -> float:
    with wave.open(str(destino), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(MUESTRAS)
        w.writeframes(b"\x00\x00" * int(segundos * MUESTRAS))
    return segundos


# --- Línea de tiempo --------------------------------------------------------
def linea_de_tiempo(trabajo: Path, carpeta_audio: Path, solo: list[str], borrador: bool, fin: float) -> dict:
    laminas = deck.cargar()
    por_id = {s["id"]: s for s in laminas}
    segs = [g for g in deck.segmentos(laminas) if not solo or g["id"] in solo]
    audio_dir = trabajo / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    cursor, eventos, pistas, capitulos, subtitulos, faltantes, avisos = ENTRADA, [], [], [], [], [], []
    for g in segs:
        origen = next((carpeta_audio / f"{g['id']}{ext}" for ext in (".mp3", ".wav", ".m4a")
                       if (carpeta_audio / f"{g['id']}{ext}").exists()), None)
        wav = audio_dir / f"{g['id']}.wav"
        if origen:
            dur = preparar_audio(origen, wav)
            sil = silencios(wav)
            # Control de calidad del audio: duración muy distinta de la esperada o silencios anormales
            razon = dur / g["segundos_estimados"]
            if not 0.8 <= razon <= 1.25:
                avisos.append(f"{g['id']}: dura {dur:.1f} s y se esperaban {g['segundos_estimados']:.1f} s (¿texto omitido o repetido?)")
            largos = [b - a for a, b in sil if b - a > 2.0]
            if largos:
                avisos.append(f"{g['id']}: tiene {len(largos)} silencio(s) de más de 2 s (el mayor, {max(largos):.1f} s)")
        elif borrador:
            dur, sil = silencio_wav(wav, g["segundos_estimados"]), []
            faltantes.append(g["id"])
        else:
            raise FileNotFoundError(f"Falta el audio de {g['id']} en {carpeta_audio} (usa --borrador para rellenar con silencio)")
        nuevo_parrafo = [not p["continua"] for p in g["pasos"]]
        inicios = alinear([p["peso"] for p in g["pasos"]], nuevo_parrafo, dur, sil)
        for j, p in enumerate(g["pasos"]):
            t0 = cursor + inicios[j]
            t1 = cursor + (inicios[j + 1] if j + 1 < len(inicios) else dur)
            s = por_id[p["lamina"]]
            if p["paso"] == 0 and s["primera_del_modulo"]:
                capitulos.append((t0 if capitulos else 0.0, deck.MODULOS[s["modulo"]]["capitulo"]))
            eventos.append(dict(t=t0, frame=f"{p['lamina']}_p{p['paso']:02d}.png", seg=g["id"]))
            subtitulos.append((t0, t1, p["texto"]))
        pistas.append((cursor, wav, dur))
        cursor += dur + deck.PAUSA_ENTRE_SEGMENTOS
    total = cursor - deck.PAUSA_ENTRE_SEGMENTOS + fin
    return dict(eventos=eventos, pistas=pistas, capitulos=capitulos, subtitulos=subtitulos, total=total,
                faltantes=faltantes, avisos=avisos, segmentos=[g["id"] for g in segs])


# --- Video y audio ----------------------------------------------------------
def escribir_audio(tl: dict, destino: Path) -> None:
    muestras = np.zeros(int(round(tl["total"] * MUESTRAS)) + MUESTRAS, dtype=np.int16)
    for inicio, wav, _ in tl["pistas"]:
        with wave.open(str(wav)) as w:
            datos = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        a = int(round(inicio * MUESTRAS))
        muestras[a:a + len(datos)] = datos
    muestras = muestras[: int(round(tl["total"] * MUESTRAS))]
    with wave.open(str(destino), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(MUESTRAS)
        w.writeframes(muestras.tobytes())


def escribir_video(tl: dict, trabajo: Path, destino: Path) -> None:
    frames, trans = trabajo / "frames", trabajo / "fundidos"
    trans.mkdir(exist_ok=True)
    total_cuadros = int(round(tl["total"] * FPS))
    cortes = [int(round(e["t"] * FPS)) for e in tl["eventos"]]
    cortes[0] = 0
    lineas = ["ffconcat version 1.0"]
    for i, e in enumerate(tl["eventos"]):
        ini = cortes[i]
        fin = cortes[i + 1] if i + 1 < len(cortes) else total_cuadros
        if fin <= ini:
            continue
        actual = frames / e["frame"]
        n_fundido = 0
        if i + 1 < len(cortes) and fin - ini > FUNDIDO * 3:
            n_fundido = FUNDIDO
        lineas += [f"file '{actual}'", f"duration {(fin - ini - n_fundido) / FPS:.6f}"]
        if n_fundido:
            a = Image.open(actual).convert("RGB")
            b = Image.open(frames / tl["eventos"][i + 1]["frame"]).convert("RGB")
            for k in range(1, n_fundido + 1):
                ruta = trans / f"f{i:04d}_{k}.png"
                Image.blend(a, b, k / (n_fundido + 1)).save(ruta, compress_level=1)
                lineas += [f"file '{ruta}'", f"duration {1 / FPS:.6f}"]
    lineas.append(f"file '{frames / tl['eventos'][-1]['frame']}'")
    lista = trabajo / "video.ffconcat"
    lista.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(lista),
            "-vf", f"fps={FPS},scale=1920:1080:out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-tune", "stillimage", "-g", str(FPS * 10),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-frames:v", str(total_cuadros), str(destino))


def _marca(t: float, coma: bool = False) -> str:
    h, resto = divmod(t, 3600)
    mi, s = divmod(resto, 60)
    if coma:
        return f"{int(h):02d}:{int(mi):02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"
    return f"{int(h)}:{int(mi):02d}:{int(s):02d}" if h else f"{int(mi):02d}:{int(s):02d}"


def _trozos(texto: str, largo: int = 84) -> list[str]:
    """Parte un párrafo en subtítulos de hasta ~2 líneas, cortando en signos de puntuación cuando se puede."""
    frases = re.findall(r"[^.?!…]+[.?!…»]*\s*", texto) or [texto]
    out = []
    for f in (x.strip() for x in frases if x.strip()):
        while len(f) > largo:
            corte = max(f.rfind(sep, 0, largo) for sep in (", ", "; ", ": ", " "))
            corte = corte if corte > 20 else largo
            out.append(f[:corte + 1].strip())
            f = f[corte + 1:].strip()
        if f:
            out.append(f)
    return out


def escribir_srt(tl: dict, destino: Path) -> None:
    bloques, n = [], 1
    for t0, t1, texto in tl["subtitulos"]:
        trozos = _trozos(texto)
        pesos = [len(x) for x in trozos]
        acum = t0
        for trozo, peso in zip(trozos, pesos):
            dur = (t1 - t0) * peso / sum(pesos)
            bloques.append(f"{n}\n{_marca(acum, True)} --> {_marca(acum + dur - 0.04, True)}\n{trozo}\n")
            acum += dur
            n += 1
    destino.write_text("\n".join(bloques), encoding="utf-8")


def ensamblar(trabajo: Path, carpeta_audio: Path, salida: Path, solo: list[str], borrador: bool, fin: float) -> dict:
    tl = linea_de_tiempo(trabajo, carpeta_audio, solo, borrador, fin)
    narracion, mudo = trabajo / "narracion.wav", trabajo / "video_mudo.mp4"
    escribir_audio(tl, narracion)
    escribir_video(tl, trabajo, mudo)
    salida.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-i", str(mudo), "-i", str(narracion), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", str(MUESTRAS),
            "-movflags", "+faststart", "-shortest", str(salida))
    base = salida.with_suffix("")
    escribir_srt(tl, Path(str(base) + ".srt"))
    Path(str(base) + "_capitulos.txt").write_text(
        "\n".join(f"{_marca(t)} {nombre}" for t, nombre in tl["capitulos"]) + "\n", encoding="utf-8")
    Path(str(base) + "_tiempos.json").write_text(json.dumps(
        dict(total=tl["total"], eventos=tl["eventos"], faltantes=tl["faltantes"]), ensure_ascii=False, indent=1), encoding="utf-8")
    return tl


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trabajo", type=Path)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--solo", nargs="*", default=[])
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--borrador", action="store_true")
    ap.add_argument("--fin", type=float, default=0.0)
    a = ap.parse_args()
    salida = a.salida or (a.trabajo / "salida" / "curso.mp4")
    tl = ensamblar(a.trabajo, a.audio, salida, a.solo, a.borrador, a.fin)
    print(f"Video: {salida}  ·  duración {_marca(tl['total'])}  ·  {len(tl['segmentos'])} audios"
          + (f"  ·  {len(tl['faltantes'])} en silencio (borrador)" if tl["faltantes"] else ""))
    for aviso in tl["avisos"]:
        print("AVISO de audio ·", aviso)
    print(f"Uso de la voz estimado a {CARACTERES_POR_SEGUNDO} caracteres por segundo; capítulos y subtítulos junto al video.")
