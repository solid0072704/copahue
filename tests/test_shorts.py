"""Pruebas del agente de Shorts (sin conexión a YouTube ni a Claude; ffmpeg solo si está instalado)."""

import json
import os
import shutil
import subprocess
from datetime import timedelta

import pytest

from agentes_shorts import descarga, editor, explorador, publicador
from agentes_shorts.config import Config
from agentes_shorts.db import BaseDatos, ahora, iso
from agentes_shorts.llm import a_segundos
from agentes_shorts.orquestador import ejecutar_ciclo, seleccionar_manual
from agentes_shorts.youtube_api import parsear_duracion, video_desde_api

HAY_FFMPEG = shutil.which("ffmpeg") is not None


def item_api(video_id="abc123xyz00", vistas=2_000_000, likes=120_000, comentarios=8_000, horas=24, duracion="PT12M30S",
             licencia="creativeCommon", categoria="27", idioma="es", canal_id="UC_demo"):
    return {
        "id": video_id,
        "snippet": {
            "title": f"Video de prueba {video_id}",
            "channelTitle": "Canal Demo",
            "channelId": canal_id,
            "publishedAt": iso(ahora() - timedelta(hours=horas)).replace("+00:00", "Z"),
            "categoryId": categoria,
            "defaultAudioLanguage": idioma,
            "liveBroadcastContent": "none",
        },
        "statistics": {"viewCount": str(vistas), "likeCount": str(likes), "commentCount": str(comentarios)},
        "contentDetails": {"duration": duracion},
        "status": {"license": licencia},
    }


class YouTubeFalso:
    """Imita la clase YouTube sin tocar la red. Cobra cuota como la real."""

    def __init__(self, db, tendencias=(), busqueda=(), fallar_subida=False):
        self.db = db
        self.cfg = Config()
        self._tendencias = [video_desde_api(i, "tendencias:CL") for i in tendencias]
        self._busqueda = [video_desde_api(i, "busqueda:cc") for i in busqueda]
        self.subidas = []
        self.fallar_subida = fallar_subida

    def cuota_disponible(self, metodo):
        from agentes_shorts.config import COSTO_CUOTA

        return self.db.cuota_usada() + COSTO_CUOTA[metodo] <= self.cfg.cuota_diaria

    def tendencias(self, region, max_resultados=50):
        self.db.sumar_cuota(1)
        return self._tendencias

    def buscar(self, **kw):
        self.db.sumar_cuota(101)
        return self._busqueda

    def videos_de_canal(self, canal, max_resultados=25):
        self.db.sumar_cuota(2)
        return []

    def detalles(self, ids, fuente=""):
        self.db.sumar_cuota(1)
        return [v for v in self._tendencias + self._busqueda if v["video_id"] in ids] or [
            video_desde_api(item_api(ids[0], licencia="youtube"), fuente)
        ]

    def subir(self, archivo, titulo, descripcion, etiquetas, **kw):
        if self.fallar_subida:
            raise RuntimeError("quotaExceeded")
        self.db.sumar_cuota(1600)
        self.subidas.append({"archivo": archivo, "titulo": titulo, "descripcion": descripcion, "etiquetas": etiquetas, **kw})
        return f"yt{len(self.subidas):03d}"


@pytest.fixture
def cfg(tmp_path):
    c = Config(db_path=str(tmp_path / "shorts.db"), carpeta_trabajo=str(tmp_path / "shorts"), regiones=["CL"], busquedas_max=1,
               videos_por_ciclo=2, cortes_por_video=2, corte_duracion_s=10, corte_min_s=6, corte_max_s=15, duracion_min_s=30,
               preset_ffmpeg="ultrafast")
    return c.normalizar()


@pytest.fixture
def db(cfg):
    base = BaseDatos(cfg.db_path)
    yield base
    base.cerrar()


# ---------------------------------------------------------------- configuración y API
def test_config_normaliza_valores_imposibles():
    c = Config(licencia="pirata", privacidad="secreto", estilo="raro", corte_max_s=900, corte_min_s=5000, programar_cada_horas=6).normalizar()
    assert c.licencia == "creativeCommon" and c.privacidad == "private" and c.estilo == "difuminado"
    assert c.corte_max_s == 180 and c.corte_min_s <= c.corte_max_s and c.corte_duracion_s <= c.corte_max_s


def test_config_desde_entorno(monkeypatch):
    monkeypatch.setenv("SHORTS_REGIONES", "mx, ar")
    monkeypatch.setenv("SHORTS_PRIVACIDAD", "public")
    monkeypatch.setenv("SHORTS_PROGRAMAR_CADA_HORAS", "4")
    monkeypatch.setenv("SHORTS_SUBTITULOS", "no")
    c = Config.desde_entorno()
    assert c.regiones == ["MX", "AR"] and c.subtitulos is False
    assert c.privacidad == "private"  # programar exige privado


def test_parsear_duracion():
    assert parsear_duracion("PT1H2M3S") == 3723
    assert parsear_duracion("PT45S") == 45
    assert parsear_duracion("P1DT1S") == 86401
    assert parsear_duracion("") == 0 and parsear_duracion(None) == 0


def test_video_desde_api():
    v = video_desde_api(item_api(), "tendencias:CL")
    assert v["video_id"] == "abc123xyz00" and v["duracion_s"] == 750 and v["vistas"] == 2_000_000
    assert v["licencia"] == "creativeCommon" and v["idioma"] == "es" and v["en_vivo"] is False


# ---------------------------------------------------------------- explorador
def test_puntuar_prefiere_crecimiento_rapido():
    rapido = video_desde_api(item_api(vistas=3_000_000, horas=12))
    lento = video_desde_api(item_api(vistas=3_000_000, horas=24 * 60, likes=10_000, comentarios=100))
    s_rapido, detalle = explorador.puntuar(rapido)
    s_lento, _ = explorador.puntuar(lento)
    assert s_rapido > s_lento and 0 <= s_lento < s_rapido <= 100
    assert any("vistas/hora" in f for f, _ in detalle)


def test_motivo_descarte(cfg):
    assert explorador.motivo_descarte(video_desde_api(item_api()), cfg) is None
    assert "Creative Commons" in explorador.motivo_descarte(video_desde_api(item_api(licencia="youtube")), cfg)
    assert "categoría" in explorador.motivo_descarte(video_desde_api(item_api(categoria="10")), cfg)
    assert "dura" in explorador.motivo_descarte(video_desde_api(item_api(duracion="PT20S")), cfg)
    assert "idioma" in explorador.motivo_descarte(video_desde_api(item_api(idioma="ja")), cfg)
    # Un canal con permiso no necesita licencia CC.
    cfg.canales = ["UC_demo"]
    assert explorador.motivo_descarte(video_desde_api(item_api(licencia="youtube")), cfg) is None
    cfg.canales, cfg.licencia = [], "cualquiera"
    assert explorador.motivo_descarte(video_desde_api(item_api(licencia="youtube")), cfg) is None


def test_explorar_selecciona_y_descarta(db, cfg):
    yt = YouTubeFalso(
        db,
        tendencias=[item_api("viral000001"), item_api("musica00001", categoria="10"), item_api("sincc000001", licencia="youtube")],
        busqueda=[item_api("ccbusq00001", vistas=800_000, horas=10)],
    )
    r = explorador.explorar(db, yt, cfg)
    assert r.revisados == 4 and r.nuevos == 4 and r.descartados == 2
    assert {v["video_id"] for v in r.seleccionados} == {"viral000001", "ccbusq00001"}
    assert db.video("musica00001")["estado"] == "descartado"
    assert db.cuota_usada() == 1 + 101
    # Segundo ciclo: nada nuevo, los seleccionados no se vuelven a procesar.
    r2 = explorador.explorar(db, yt, cfg)
    assert r2.nuevos == 0 and r2.seleccionados == []


def test_recolectar_respeta_cuota(db, cfg):
    db.sumar_cuota(cfg.cuota_diaria - 50)
    yt = YouTubeFalso(db, tendencias=[item_api()], busqueda=[item_api("ccbusq00001")])
    candidatos, errores = explorador.recolectar(yt, cfg)
    assert len(candidatos) == 1 and any("sin cuota" in e for e in errores)


# ---------------------------------------------------------------- editor: selección de momentos
def heatmap_sintetico(duracion=300, pico=(120, 150)):
    tramo = duracion / 100
    return [
        {"start_time": i * tramo, "end_time": (i + 1) * tramo, "value": 1.0 if pico[0] <= i * tramo < pico[1] else 0.1}
        for i in range(100)
    ]


def test_ventanas_por_heatmap_encuentra_el_pico():
    ventanas = editor.ventanas_por_heatmap(heatmap_sintetico(), 300, 30, n=2)
    assert len(ventanas) == 2
    ini, fin, valor = ventanas[0] if ventanas[0][2] >= ventanas[1][2] else ventanas[1]
    assert 115 <= ini <= 125 and fin - ini == 30 and valor == 1.0
    # Las dos ventanas no se solapan.
    (a, b, _), (c, d, _) = sorted(ventanas)
    assert b <= c


def test_ventanas_por_heatmap_vacio_o_video_corto():
    assert editor.ventanas_por_heatmap([], 300, 30) == []
    assert editor.ventanas_por_heatmap(heatmap_sintetico(), 20, 30) == []


def test_ventanas_por_audio_y_uniformes():
    niveles = [-40.0] * 120
    for t in range(70, 90):
        niveles[t] = -10.0
    ventanas = editor.ventanas_por_audio(niveles, 120, 20, n=1)
    assert len(ventanas) == 1 and 65 <= ventanas[0][0] <= 75
    uniformes = editor.ventanas_uniformes(300, 30, n=3)
    assert len(uniformes) == 3 and all(f - i == 30 for i, f, _ in uniformes)
    assert uniformes[0][0] < uniformes[1][0] < uniformes[2][0] and uniformes[-1][1] <= 300
    assert editor.ventanas_uniformes(10, 30) == [(0.0, 10.0, 0.0)]


def test_elegir_cortes_sin_ia_usa_heatmap(cfg):
    video = {"video_id": "viral000001", "titulo": "Prueba", "duracion_s": 300}
    material = {"duracion_s": 300, "heatmap": heatmap_sintetico(), "subtitulos": [(120.0, 125.0, "hola"), (126.0, 130.0, "mundo")]}
    cortes = editor.elegir_cortes(cfg, video, material)
    assert len(cortes) == 2 and cortes[0]["metodo"] == "heatmap"
    assert all(cfg.corte_min_s <= c["fin_s"] - c["inicio_s"] <= cfg.corte_max_s for c in cortes)
    assert cortes[0]["indice"] == 1 and cortes[1]["indice"] == 2 and cortes[0]["fin_s"] <= cortes[1]["inicio_s"]
    assert "hola mundo" in " ".join(c["texto"] for c in cortes)


def test_elegir_cortes_con_ia_falsa(cfg):
    class IA:
        def elegir_cortes(self, video, transcripcion, pistas, n, duracion_video):
            assert "[0:00]" in transcripcion and n == 2 and duracion_video == 300
            return [
                {"inicio_s": 10, "fin_s": 22, "titulo": "El dato que nadie esperaba", "descripcion": "Contexto.", "etiquetas": ["datos"]},
                {"inicio_s": 15, "fin_s": 27, "titulo": "Se solapa con el anterior", "descripcion": "", "etiquetas": []},
                {"inicio_s": 200, "fin_s": 400, "titulo": "Demasiado largo", "descripcion": "", "etiquetas": []},
            ]

        def titular(self, video, texto):
            return {"titulo": "Título IA", "descripcion": "d", "etiquetas": ["x"]}

    material = {"duracion_s": 300, "heatmap": [], "subtitulos": [(float(t), float(t + 2), f"palabra{t}") for t in range(0, 300, 2)]}
    cortes = editor.elegir_cortes(cfg, {"video_id": "v", "titulo": "T", "duracion_s": 300}, material, IA())
    assert [c["metodo"] for c in cortes] == ["ia", "ia"]
    assert cortes[0]["titulo"] == "El dato que nadie esperaba"
    assert cortes[1]["inicio_s"] == 200 and cortes[1]["fin_s"] - cortes[1]["inicio_s"] == cfg.corte_max_s


def test_a_segundos():
    assert a_segundos("1:05") == 65 and a_segundos("1:02:03") == 3723 and a_segundos("75") == 75
    assert a_segundos("x") is None and a_segundos("1:2:3:4") is None


# ---------------------------------------------------------------- subtítulos
JSON3 = {
    "events": [
        {"tStartMs": 0, "dDurationMs": 5000, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 500, "dDurationMs": 3000, "segs": [{"utf8": "Hola"}, {"utf8": " a", "tOffsetMs": 400}, {"utf8": " todos", "tOffsetMs": 900}]},
        {"tStartMs": 3200, "dDurationMs": 10, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 3200, "dDurationMs": 2800, "segs": [{"utf8": "bienvenidos al video"}]},
        {"tStartMs": 3200, "dDurationMs": 2800, "segs": [{"utf8": "bienvenidos al video"}]},
    ]
}


def test_parsear_json3_y_srt(tmp_path):
    segs = descarga.parsear_json3(JSON3)
    assert segs == [(0.5, 3.2, "Hola a todos"), (3.2, 6.0, "bienvenidos al video")]
    ruta = tmp_path / "c.srt"
    assert editor.escribir_srt(segs, 2.0, 6.0, str(ruta))
    contenido = ruta.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,200\nHola a todos" in contenido
    assert "00:00:01,200 --> 00:00:04,000\nbienvenidos al video" in contenido
    assert editor.texto_ventana(segs, 3.0, 4.0) == "Hola a todos bienvenidos al video"
    assert editor.transcripcion_para_ia(segs).startswith("[0:00] Hola a todos")


def test_cargar_subtitulos_elige_idioma(tmp_path):
    (tmp_path / "vid.en.json3").write_text(json.dumps(JSON3), encoding="utf-8")
    (tmp_path / "vid.es.json3").write_text(json.dumps({"events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "Hola"}]}]}), encoding="utf-8")
    assert descarga.cargar_subtitulos(str(tmp_path), "vid", ["es"])[0][2] == "Hola"
    assert descarga.cargar_subtitulos(str(tmp_path), "vid", ["fr"])[0][2] == "Hola"  # cae a es antes que en
    assert descarga.cargar_subtitulos(str(tmp_path), "otro", ["es"]) == []


# ---------------------------------------------------------------- publicador
def test_armar_metadatos(cfg):
    video = {"video_id": "viral000001", "titulo": "Un título muy largo " * 6, "canal": "Canal Demo", "licencia": "creativeCommon"}
    corte = {"indice": 2, "inicio_s": 61, "fin_s": 106, "titulo": "", "descripcion": "", "etiquetas": ["finanzas", "#ahorro"]}
    titulo, descripcion, etiquetas = publicador.armar_metadatos(cfg, corte, video)
    assert len(titulo) <= 100 and titulo.endswith("#Shorts") and "parte 2" in titulo
    assert "https://youtu.be/viral000001" in descripcion and "Creative Commons" in descripcion and "1:01 a 1:46" in descripcion
    assert etiquetas == ["finanzas", "ahorro", "Shorts"] and "#finanzas #ahorro #Shorts" in descripcion
    corte["titulo"] = "Título propio"
    assert publicador.armar_metadatos(cfg, corte, video)[0] == "Título propio #Shorts"


def test_publicar_respeta_limite_y_cuota(db, cfg, tmp_path):
    yt = YouTubeFalso(db)
    archivo = tmp_path / "corte.mp4"
    archivo.write_bytes(b"0" * 10)
    db.guardar_video(video_desde_api(item_api("viral000001")))
    for i in (1, 2, 3):
        db.guardar_corte({"video_id": "viral000001", "indice": i, "inicio_s": i * 10, "fin_s": i * 10 + 8, "archivo": str(archivo)})
    cfg.max_subidas_dia = 2
    r = publicador.publicar_pendientes(db, yt, cfg)
    assert len(r.subidos) == 2 and r.omitidos and "máximo diario" in r.omitidos[0]
    assert db.subidos_hoy() == 2 and db.cuota_usada() == 3200
    assert yt.subidas[0]["privacidad"] == "private" and yt.subidas[0]["publicar_en"] is None
    assert db.video("viral000001")["estado"] != "publicado"  # queda uno pendiente
    # Sin cuota no se intenta subir.
    cfg.max_subidas_dia = 10
    db.sumar_cuota(cfg.cuota_diaria)
    r = publicador.publicar_pendientes(db, yt, cfg)
    assert not r.subidos and "cuota" in r.omitidos[0]


def test_publicar_programado_y_errores(db, cfg, tmp_path):
    archivo = tmp_path / "corte.mp4"
    archivo.write_bytes(b"0" * 10)
    db.guardar_video(video_desde_api(item_api("viral000001")))
    db.guardar_corte({"video_id": "viral000001", "indice": 1, "inicio_s": 0, "fin_s": 8, "archivo": str(archivo)})
    db.guardar_corte({"video_id": "viral000001", "indice": 2, "inicio_s": 20, "fin_s": 28, "archivo": str(archivo)})
    db.guardar_corte({"video_id": "viral000001", "indice": 3, "inicio_s": 40, "fin_s": 48, "archivo": str(tmp_path / "no_existe.mp4")})
    cfg.programar_cada_horas = 6
    cfg.normalizar()
    yt = YouTubeFalso(db)
    r = publicador.publicar_pendientes(db, yt, cfg)
    assert len(r.subidos) == 2 and len(r.errores) == 1
    fechas = [s["publicar_en"] for s in yt.subidas]
    assert all(f.endswith("Z") for f in fechas) and fechas[1] > fechas[0]
    assert db.cortes(estado="error")[0]["error"].startswith("el archivo")
    assert db.video("viral000001")["estado"] == "publicado"
    # Un fallo de la API deja el corte en error sin romper el ciclo.
    db.guardar_corte({"video_id": "viral000001", "indice": 4, "inicio_s": 60, "fin_s": 68, "archivo": str(archivo)})
    r = publicador.publicar_pendientes(db, YouTubeFalso(db, fallar_subida=True), cfg)
    assert r.errores and db.cortes(estado="error", video_id="viral000001")


def test_simular_no_sube(db, cfg, tmp_path):
    archivo = tmp_path / "corte.mp4"
    archivo.write_bytes(b"0")
    db.guardar_video(video_desde_api(item_api("viral000001")))
    db.guardar_corte({"video_id": "viral000001", "indice": 1, "inicio_s": 0, "fin_s": 8, "archivo": str(archivo)})
    yt = YouTubeFalso(db)
    r = publicador.publicar_pendientes(db, yt, cfg, simular=True)
    assert r.subidos and not yt.subidas and db.cortes(estado="renderizado")


# ---------------------------------------------------------------- ciclo completo
def material_falso(cfg, video_id, idiomas=None, archivo=None):
    return {
        "archivo": archivo or "",
        "heatmap": heatmap_sintetico(duracion=120, pico=(30, 45)),
        "subtitulos": [(float(t), float(t + 2), f"frase {t}") for t in range(0, 120, 2)],
        "duracion_s": 120.0,
        "titulo": "Video descargado",
        "idioma": "es",
    }


def render_falso(cfg, origen, inicio, fin, destino, rotulo=None, subtitulos=None):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "wb") as f:
        f.write(b"mp4")
    return destino


def test_ciclo_completo_con_dobles(db, cfg):
    cfg.borrar_original = False
    yt = YouTubeFalso(db, tendencias=[item_api("viral000001", duracion="PT2M"), item_api("viral000002", duracion="PT2M", vistas=500_000)])
    r = ejecutar_ciclo(db, cfg, yt, usar_ia=False, descargar_fn=material_falso, renderizar_fn=render_falso)
    assert len(r.exploracion.seleccionados) == 2 and len(r.cortes) == 4
    assert len(r.publicacion.subidos) == 3  # máximo diario por defecto
    assert db.video("viral000001")["estado"] == "publicado"
    assert db.video("viral000002")["estado"] == "cortado"
    assert yt.subidas[0]["titulo"].endswith("#Shorts") and "https://youtu.be/viral000001" in yt.subidas[0]["descripcion"]
    assert not r.errores
    # Siguiente ciclo: el pendiente no sube (cupo del día agotado) y no se recorta nada nuevo.
    r2 = ejecutar_ciclo(db, cfg, yt, usar_ia=False, descargar_fn=material_falso, renderizar_fn=render_falso)
    assert r2.cortes == [] and r2.publicacion.omitidos
    assert "Explorador" in r2.texto() and "Publicador" in r2.texto()


def test_ciclo_tolera_descarga_fallida(db, cfg):
    def descarga_rota(cfg, video_id, idiomas=None):
        raise RuntimeError("Sign in to confirm you're not a bot")

    yt = YouTubeFalso(db, tendencias=[item_api("viral000001", duracion="PT2M")])
    r = ejecutar_ciclo(db, cfg, yt, usar_ia=False, descargar_fn=descarga_rota, renderizar_fn=render_falso)
    assert r.cortes == [] and any("not a bot" in e for e in r.errores)
    assert db.video("viral000001")["estado"] == "error"


def test_seleccionar_manual(db, cfg):
    yt = YouTubeFalso(db)
    v = seleccionar_manual(db, yt, cfg, "manual00001")
    assert v["estado"] == "seleccionado" and v["licencia"] == "youtube"


# ---------------------------------------------------------------- ffmpeg real
@pytest.mark.skipif(not HAY_FFMPEG, reason="ffmpeg no está instalado")
def test_renderizar_genera_short_vertical(cfg, tmp_path):
    origen = tmp_path / "fuente.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
         "-t", "12", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(origen)],
        check=True,
    )
    subtitulos = [(3.0, 5.0, "Hola, esto es una prueba"), (5.5, 8.0, "con acentos áéíóú y ñ")]
    destino = str(tmp_path / "salida" / "corte_01.mp4")
    for estilo in ("difuminado", "recorte"):
        cfg.estilo = estilo
        salida = editor.renderizar(cfg, str(origen), 2.0, 9.0, destino, rotulo="Título: ¿funciona 100%?", subtitulos=subtitulos)
        info = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-show_entries", "format=duration", "-of", "json", salida],
            capture_output=True, text=True, check=True,
        ).stdout
        datos = json.loads(info)
        assert datos["streams"][0]["width"] == 1080 and datos["streams"][0]["height"] == 1920
        assert 6.5 <= float(datos["format"]["duration"]) <= 7.6
    assert os.path.exists(str(tmp_path / "salida" / "corte_01.srt"))


@pytest.mark.skipif(not HAY_FFMPEG, reason="ffmpeg no está instalado")
def test_energia_audio_detecta_el_tramo_fuerte(tmp_path):
    origen = tmp_path / "audio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "20",
         "-af", "volume='if(between(t,8,12),1,0.05)':eval=frame", "-c:a", "aac", str(origen)],
        check=True,
    )
    niveles = editor.energia_audio(str(origen))
    assert 18 <= len(niveles) <= 21
    ventanas = editor.ventanas_por_audio(niveles, 20, 4, n=1, margen=0)
    assert ventanas and 7 <= ventanas[0][0] <= 9
