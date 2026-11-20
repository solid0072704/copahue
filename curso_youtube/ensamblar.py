"""Une las láminas con la narración y produce el video (MP4 1080p), los capítulos y los subtítulos.

Uso:
    python -m curso_youtube.ensamblar <trabajo> <carpeta_audio> [--solo seg_000 seg_001 ...]
                                      [--salida video.mp4] [--borrador] [--fin 20] [--anexos]

- Espera un archivo de audio por segmento: <carpeta_audio>/seg_XXX.mp3 (o .wav). Los nombres y el texto de
  cada segmento salen de `python -m curso_youtube.guion --voz`.
- Las imágenes deben existir en <trabajo>/frames (se generan con `python -m curso_youtube.construir`).
- `--borrador` reemplaza los audios que faltan por silencio de la duración estimada, para revisar el ritmo.
- `--fin` agrega segundos de la última lámina al final (espacio para la pantalla final de YouTube).
- `--anexos` no renderiza: vuelve a escribir los subtítulos, los capítulos y los tiempos junto al video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import deck
from .voz import CARACTERES_POR_SEGUNDO, a_voz

FPS = 30
MUESTRAS = 48000
VOLUMEN_OBJETIVO = -16.0   # LUFS
ENTRADA = 0.35             # segundos antes de la primera palabra
FUNDIDO = 7                # cuadros de fundido entre una lámina y la siguiente
BARRIDO = 17               # cuadros del barrido con que entran las portadas
SOMBRA_BARRIDO = 150       # ancho, en pixeles, de la sombra que acompaña al barrido
ABREN_CON_BARRIDO = {"seccion", "titulo_curso"}
ORO = (227, 166, 47)
CALIDAD_JPEG = 95


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
    cursor, eventos, pistas, capitulos, subtitulos, faltantes, avisos, pausas = ENTRADA, [], [], [], [], [], [], []
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
        pausas += [(cursor + a, cursor + b) for a, b in sil]
        pistas.append((cursor, wav, dur))
        cursor += dur + deck.PAUSA_ENTRE_SEGMENTOS
    total = cursor - deck.PAUSA_ENTRE_SEGMENTOS + fin
    return dict(eventos=eventos, pistas=pistas, capitulos=capitulos, subtitulos=subtitulos, total=total,
                faltantes=faltantes, avisos=avisos, pausas=pausas, segmentos=[g["id"] for g in segs])


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


def _suave(p: float) -> float:
    return 4 * p ** 3 if p < .5 else 1 - (-2 * p + 2) ** 3 / 2


def _jpeg(img: Image.Image, ruta: Path) -> Path:
    img.save(ruta, quality=CALIDAD_JPEG, subsampling=2)
    return ruta


def _barrer(a: Image.Image, b: Image.Image, p: float) -> Image.Image:
    """Barrido de izquierda a derecha: `b` entra sobre `a`, con un filete dorado en el borde y su sombra."""
    ancho, alto = b.size
    x = round(ancho * p)
    img = a.copy()
    sombra = Image.linear_gradient("L").rotate(90).resize((SOMBRA_BARRIDO, alto)).point(lambda v: v * 120 // 255)
    img.paste((4, 8, 14), (x, 0), sombra)
    if x > 0:
        img.paste(b.crop((0, 0, x, alto)), (0, 0))
    ImageDraw.Draw(img).rectangle((x - 2, 0, x + 1, alto), fill=ORO)
    return img


def secuencia(tl: dict, trabajo: Path) -> list[tuple[Path, int, str]]:
    """El video completo como lista de (imagen JPEG, cuadros que dura, lámina).

    Cada paso muestra primero los cuadros de su animación (si se capturaron con `construir --animar`) y después
    su imagen final hasta que empieza el siguiente. Al cambiar de lámina, los primeros cuadros se funden con la
    imagen con que terminó la anterior; las portadas de capítulo entran con un barrido. Las imágenes que se
    generan aquí (fundidos, barridos) no se vuelven a escribir si sus fuentes no cambiaron.
    """
    frames, trans = trabajo / "frames", trabajo / "fundidos"
    trans.mkdir(exist_ok=True)
    manifiesto = trabajo / "html" / "laminas.json"
    tipos = {s["id"]: s.get("tipo", "") for s in json.loads(manifiesto.read_text())} if manifiesto.exists() else {}
    animadas: dict[str, dict] = {}

    def cuadros(lamina: str, clave: str) -> int:
        """Cuántos cuadros de animación tiene un paso (0 si la lámina no se capturó con animación)."""
        if lamina not in animadas:
            marca = frames / f"{lamina}.listo"
            animadas[lamina] = json.loads(marca.read_text())["cuadros"] if marca.exists() else {}
        return animadas[lamina].get(clave, 0)

    def fija(clave: str) -> Path:
        """La imagen final de un paso en JPEG: el video se arma con un solo formato de imagen."""
        jpg, png = frames / f"{clave}.jpg", frames / f"{clave}.png"
        if jpg.exists() and (not png.exists() or jpg.stat().st_mtime >= png.stat().st_mtime):
            return jpg
        destino = trans / f"{clave}.jpg"
        if not destino.exists() or destino.stat().st_mtime < png.stat().st_mtime:
            _jpeg(Image.open(png).convert("RGB"), destino)
        return destino

    total_cuadros = int(round(tl["total"] * FPS))
    cortes = [int(round(e["t"] * FPS)) for e in tl["eventos"]]
    cortes[0] = 0
    seq: list[tuple[Path, int, str]] = []
    previa: Path | None = None
    lamina_previa = ""
    for i, e in enumerate(tl["eventos"]):
        n = (cortes[i + 1] if i + 1 < len(cortes) else total_cuadros) - cortes[i]
        if n <= 0:
            continue
        clave = e["frame"][:-4]
        lamina = clave.rsplit("_p", 1)[0]
        final = fija(clave)
        con_animacion = min(cuadros(lamina, clave), n)
        imagenes = [frames / f"{clave}_f{f:03d}.jpg" for f in range(con_animacion)]
        cambia = lamina != lamina_previa
        if previa is not None and (cambia or not con_animacion):
            # transición desde la imagen anterior, sobre los primeros cuadros de la animación (o sobre la imagen final)
            barrido = cambia and tipos.get(lamina) in ABREN_CON_BARRIDO
            largo = min(BARRIDO if barrido else FUNDIDO if con_animacion else 4, max(0, n - 2))
            a = None
            for k in range(largo):
                fuente = imagenes[k] if k < len(imagenes) else final
                ruta = trans / f"{clave}_desde_{previa.stem}_t{k:02d}.jpg"
                if not ruta.exists() or ruta.stat().st_mtime < max(previa.stat().st_mtime, fuente.stat().st_mtime):
                    a = a or Image.open(previa).convert("RGB")
                    b = Image.open(fuente).convert("RGB")
                    peso = (k + 1) / (largo + 1)
                    _jpeg(_barrer(a, b, _suave(peso)) if barrido else Image.blend(a, b, peso * peso * (3 - 2 * peso)), ruta)
                if k < len(imagenes):
                    imagenes[k] = ruta
                else:
                    imagenes.append(ruta)
        seq += [(img, 1, lamina) for img in imagenes]
        if n - len(imagenes) > 0:
            seq.append((final, n - len(imagenes), lamina))
        previa, lamina_previa = final, lamina
    return seq


def _codificar(lista: list[tuple[Path, int]], destino: Path) -> None:
    cuadros = sum(n for _, n in lista)
    lineas = ["ffconcat version 1.0"]
    for img, n in lista:
        lineas += [f"file '{img}'", f"duration {n / FPS:.6f}"]
    lineas.append(f"file '{lista[-1][0]}'")
    concat = destino.with_suffix(".ffconcat")
    concat.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    # Las imágenes son JPEG (BT.601, rango completo): se pasan por RGB para llegar bien a BT.709 de rango de video.
    filtro = (f"fps={FPS},scale=1920:1080:in_color_matrix=bt601:in_range=pc,format=gbrp,"
              "scale=1920:1080:out_color_matrix=bt709:out_range=tv,format=yuv420p")
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat), "-vf", filtro,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", str(FPS * 10), "-bf", "2",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-frames:v", str(cuadros), str(destino))


def escribir_video(tl: dict, trabajo: Path, destino: Path) -> None:
    """Codifica el video por tramos (uno por módulo) y los une sin recodificar.

    Un tramo ya codificado con las mismas imágenes no se repite: si el proceso se interrumpe, se retoma donde quedó.
    """
    seq = secuencia(tl, trabajo)
    tramos_dir = trabajo / "tramos"
    tramos_dir.mkdir(exist_ok=True)
    grupos: list[list[tuple[Path, int]]] = []
    modulo = None
    for img, n, lamina in seq:
        m = lamina.split("_")[0]
        if m != modulo:
            grupos.append([])
            modulo = m
        grupos[-1].append((img, n))
    partes = []
    for k, grupo in enumerate(grupos):
        parte = tramos_dir / f"tramo_{k:02d}.mp4"
        firma = hashlib.sha1("|".join(f"{img}:{n}:{img.stat().st_mtime_ns}" for img, n in grupo).encode()).hexdigest()
        marca = parte.with_suffix(".firma")
        if not (parte.exists() and marca.exists() and marca.read_text() == firma):
            marca.unlink(missing_ok=True)
            _codificar(grupo, parte)
            marca.write_text(firma)
        partes.append(parte)
    lista = tramos_dir / "tramos.txt"
    lista.write_text("".join(f"file '{x}'\n" for x in partes), encoding="utf-8")
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(lista), "-c", "copy", str(destino))


def _marca(t: float, coma: bool = False) -> str:
    h, resto = divmod(t, 3600)
    mi, s = divmod(resto, 60)
    if coma:
        return f"{int(h):02d}:{int(mi):02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"
    return f"{int(h)}:{int(mi):02d}:{int(s):02d}" if h else f"{int(mi):02d}:{int(s):02d}"


LARGO_SUBTITULO = 84        # caracteres: dos líneas de 42
_FRASE = re.compile(r".+?(?:[.?!…]+[»\"”)]*(?=\s+|$)|$)", re.S)   # el punto de «6.105» no termina una frase


_ENLACES = {"y", "e", "o", "u", "ni", "que", "de", "del", "con", "en", "a", "al", "para", "por", "sin", "desde", "hasta",
            "porque", "pero", "como", "cuando", "donde", "si", "sino", "entre", "sobre", "contra", "según", "aunque"}
_COLGANTES = _ENLACES | {"el", "la", "los", "las", "un", "una", "unos", "unas", "su", "sus", "tu", "tus", "mi", "mis",
                         "lo", "le", "les", "se", "te", "no", "más", "muy", "es", "son", "cada", "este", "esta"}


def _ajuste(t: str, i: int) -> float:
    """Qué tan natural es cortar en el espacio i: mejor después de un signo o antes de una preposición o conjunción;
    peor si deja una palabra colgando o separa una cifra de lo que mide."""
    antes = t[:i].rsplit(" ", 1)[-1].lower()
    despues = t[i + 1:].split(" ", 1)[0].lower().strip("¿¡«\"(")
    if antes[-1:] in ",;:.?!…":
        return -14.0
    ajuste = 0.0
    if despues in _ENLACES:
        ajuste -= 7
    if antes.strip("¿¡«\"(") in _COLGANTES:
        ajuste += 9
    if re.fullmatch(r"[\d.,]+", antes):
        ajuste += 12
    if despues.startswith("%"):
        ajuste += 20
    return ajuste


def _cortar(frase: str, largo: int) -> list[str]:
    """Parte una frase larga en trozos parejos, en el lugar más natural cerca del largo buscado."""
    if len(frase) <= largo:
        return [frase]
    partes = -(-len(frase) // largo)
    objetivo = len(frase) / partes
    espacios = [i for i, c in enumerate(frase) if c == " " and 12 <= i <= largo and len(frase) - i > 12]
    if not espacios:
        return [frase]
    corte = min(espacios, key=lambda i: abs(i - objetivo) + _ajuste(frase, i))
    return [frase[:corte]] + _cortar(frase[corte + 1:], largo)


def _trozos(texto: str, largo: int = LARGO_SUBTITULO) -> list[str]:
    """Parte un párrafo en subtítulos de hasta dos líneas: una frase por subtítulo, las muy cortas se juntan."""
    frases = [f.strip() for f in _FRASE.findall(" ".join(texto.split())) if f.strip()]
    out: list[str] = []
    for f in frases:
        if out and len(out[-1]) + 1 + len(f) <= largo and min(len(out[-1]), len(f)) < 28:
            out[-1] += " " + f
        else:
            out += _cortar(f, largo)
    return out


def _dos_lineas(t: str, ancho: int = LARGO_SUBTITULO // 2) -> str:
    """Si no cabe en una línea, la divide en dos de largo parecido, en el lugar más natural."""
    if len(t) <= ancho:
        return t
    espacios = [i for i, c in enumerate(t) if c == " "]
    if not espacios:
        return t
    parejos = [i for i in espacios if max(i, len(t) - i - 1) <= ancho + 4] or espacios
    corte = min(parejos, key=lambda i: abs(i - len(t) / 2) + _ajuste(t, i))
    return t[:corte] + "\n" + t[corte + 1:]


def _fuerza(trozo_anterior: str) -> int:
    """Pausa que se espera después de un trozo: 2 si cierra una frase (o termina en dos puntos), 1 si termina en
    coma o punto y coma, 0 si no hay signo."""
    ultimo = trozo_anterior.rstrip("»\"”)")[-1:]
    return 2 if ultimo in ".?!…:" else 1 if ultimo in ",;" else 0


_ESPERA = (0, 3, 9)   # pausa antes de un trozo según la fuerza del corte, medida en caracteres hablados


def _peso_hablado(trozo: str) -> float:
    """Largo del texto tal como se dice, más las pausas de su puntuación interior."""
    return (len(a_voz(trozo)) + _ESPERA[2] * len(re.findall(r"[.?!…:]\s", trozo))
            + _ESPERA[1] * len(re.findall(r"[,;]\s", trozo)))


def tiempos_de_subtitulos(t0: float, t1: float, trozos: list[str], pausas: list[tuple[float, float]]) -> list[float]:
    """Instante en que empieza cada subtítulo de un paso que va de t0 a t1.

    Estima cada cambio según el largo del texto hablado y, si el trozo anterior termina en un signo de puntuación,
    lo lleva a la pausa real más cercana. Cada ajuste corrige la estimación de los que siguen.
    """
    n = len(trozos)
    if n == 1:
        return [t0]
    fuerza = [0] + [_fuerza(trozos[i - 1]) for i in range(1, n)]
    # posición, en caracteres hablados, en que empieza a sonar cada trozo
    pos, corrido = [], 0.0
    for f, trozo in zip(fuerza, trozos):
        pos.append(corrido + _ESPERA[f])
        corrido += _ESPERA[f] + _peso_hablado(trozo)
    total = corrido
    # la voz termina antes de t1 si el paso cierra con una pausa
    fin_voz = min([a for a, b in pausas if t0 + 0.3 < a < t1 and b >= t1 - 0.12] + [t1])
    dentro = [(a, b) for a, b in pausas if a > t0 + 0.2 and b < fin_voz - 0.2]
    inicios: list[float | None] = [t0] + [None] * (n - 1)
    ancla = 0
    for i in range(1, n):
        if not fuerza[i]:
            continue
        ritmo = (fin_voz - inicios[ancla]) / (total - pos[ancla])
        estimado = inicios[ancla] + (pos[i] - pos[ancla]) * ritmo
        margen, tope, premio = (1.0, 0.8, 2.5) if fuerza[i] == 2 else (0.6, 0.4, 1.0)
        mejor, nota = None, -1e9
        for a, b in dentro:
            retoma = b - 0.06
            if retoma < inicios[ancla] + 0.4 or abs(retoma - estimado) > margen:
                continue
            valor = premio * min(b - a, tope) - 0.8 * abs(retoma - estimado)
            if valor > nota:
                mejor, nota = (a, b), valor
        if mejor:
            inicios[i] = mejor[1] - 0.06
            dentro = [p for p in dentro if p[0] > mejor[0]]
            ancla = i
    # los cambios sin pausa se reparten, según el texto hablado, entre los que sí quedaron fijos
    fijos = [i for i, x in enumerate(inicios) if x is not None]
    for k, i0 in enumerate(fijos):
        i1 = fijos[k + 1] if k + 1 < len(fijos) else n
        t_fin, p_fin = (inicios[i1], pos[i1]) if i1 < n else (fin_voz, total)
        for j in range(i0 + 1, i1):
            inicios[j] = inicios[i0] + (t_fin - inicios[i0]) * (pos[j] - pos[i0]) / (p_fin - pos[i0])
    return [float(x) for x in inicios]


def subtitulos_con_tiempo(tl: dict) -> list[tuple[float, float, str]]:
    """(inicio, fin, texto) de cada subtítulo del video, en orden y sin traslapes."""
    out = []
    pausas = tl.get("pausas", [])
    for t0, t1, texto in tl["subtitulos"]:
        trozos = _trozos(texto)
        cerca = [p for p in pausas if p[1] > t0 and p[0] < t1]
        inicios = tiempos_de_subtitulos(t0, t1, trozos, cerca)
        for i, trozo in enumerate(trozos):
            fin = inicios[i + 1] if i + 1 < len(trozos) else t1
            out.append((inicios[i], min(fin, max(inicios[i] + 0.3, fin - 0.04)), trozo))
    return out


def escribir_srt(tl: dict, destino: Path) -> None:
    bloques = [f"{n}\n{_marca(a, True)} --> {_marca(b, True)}\n{_dos_lineas(texto)}\n"
               for n, (a, b, texto) in enumerate(subtitulos_con_tiempo(tl), 1)]
    destino.write_text("\n".join(bloques), encoding="utf-8")


def ensamblar(trabajo: Path, carpeta_audio: Path, salida: Path, solo: list[str], borrador: bool, fin: float) -> dict:
    tl = linea_de_tiempo(trabajo, carpeta_audio, solo, borrador, fin)
    narracion, mudo = trabajo / "narracion.wav", trabajo / "video_mudo.mp4"
    escribir_audio(tl, narracion)
    escribir_video(tl, trabajo, mudo)
    salida.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-i", str(mudo), "-i", str(narracion), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", str(MUESTRAS),
            "-movflags", "+faststart", "-shortest", str(salida))
    escribir_anexos(tl, salida)
    return tl


def escribir_anexos(tl: dict, salida: Path) -> None:
    """Subtítulos, capítulos y tiempos de cada imagen, junto al video."""
    base = salida.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    escribir_srt(tl, Path(str(base) + ".srt"))
    Path(str(base) + "_capitulos.txt").write_text(
        "\n".join(f"{_marca(t)} {nombre}" for t, nombre in tl["capitulos"]) + "\n", encoding="utf-8")
    Path(str(base) + "_tiempos.json").write_text(json.dumps(
        dict(total=tl["total"], eventos=tl["eventos"], faltantes=tl["faltantes"]), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trabajo", type=Path)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--solo", nargs="*", default=[])
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--borrador", action="store_true")
    ap.add_argument("--fin", type=float, default=0.0)
    ap.add_argument("--anexos", action="store_true",
                    help="no renderiza: solo vuelve a escribir subtítulos, capítulos y tiempos junto al video")
    a = ap.parse_args()
    salida = a.salida or (a.trabajo / "salida" / "curso.mp4")
    if a.anexos:
        tl = linea_de_tiempo(a.trabajo, a.audio, a.solo, a.borrador, a.fin)
        escribir_anexos(tl, salida)
    else:
        tl = ensamblar(a.trabajo, a.audio, salida, a.solo, a.borrador, a.fin)
    print(f"Video: {salida}  ·  duración {_marca(tl['total'])}  ·  {len(tl['segmentos'])} audios"
          + (f"  ·  {len(tl['faltantes'])} en silencio (borrador)" if tl["faltantes"] else ""))
    for aviso in tl["avisos"]:
        print("AVISO de audio ·", aviso)
    print(f"Uso de la voz estimado a {CARACTERES_POR_SEGUNDO} caracteres por segundo; capítulos y subtítulos junto al video.")
