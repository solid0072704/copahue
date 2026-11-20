"""Integridad del curso: cada lámina tiene narración y ningún elemento queda sin aparecer."""

import re

from curso_youtube import deck
from curso_youtube.laminas import pagina
from curso_youtube.voz import a_voz


def _laminas():
    return deck.cargar()


def test_cada_lamina_tiene_narracion_y_id_unico():
    laminas = _laminas()
    assert len(laminas) > 90
    assert len({s["id"] for s in laminas}) == len(laminas)
    for s in laminas:
        assert s["pasos"] and all(isinstance(p, str) and p.strip() for p in s["pasos"]), s["id"]


def test_ningun_elemento_aparece_despues_del_ultimo_paso():
    for s in _laminas():
        ctx = dict(kicker=s["kicker"], modulo=s["modulo"], n_modulos=s["n_modulos"], avance=s["avance"])
        html = pagina(s, ctx)
        usados = set()
        for valor in re.findall(r'data-(?:p|foco)="([0-9,]+)"', html):
            usados.update(int(v) for v in valor.split(","))
        assert not usados or max(usados) <= len(s["pasos"]) - 1, f"{s['id']}: paso {max(usados)} de {len(s['pasos'])}"


def test_texto_de_voz_sin_cifras_ni_simbolos():
    for g in deck.segmentos():
        assert not re.search(r"\d", g["voz"]), f"{g['id']} tiene cifras sin convertir: {g['voz'][:80]}"
        assert not re.search(r"[%«»×÷→]", g["voz"]), f"{g['id']} tiene símbolos que la voz no lee"
        assert "Claude" not in g["voz"]
        assert g["caracteres"] <= 1500, f"{g['id']} es demasiado largo para un solo audio"


def test_duracion_cercana_a_una_hora():
    segs = deck.segmentos()
    minutos = sum(g["segundos_estimados"] + deck.PAUSA_ENTRE_SEGMENTOS for g in segs) / 60
    assert 58 <= minutos <= 68, f"duración estimada: {minutos:.1f} min"


def test_el_indice_cubre_todos_los_capitulos():
    """La lámina de índice nombra cada capítulo que viene después una sola vez, con sus minutos."""
    laminas = _laminas()
    indice = next(s for s in laminas if s["tipo"] == "indice")
    assert indice["modulo"] == 0
    claves = [m["clave"] for m in deck.MODULOS]
    cubiertos = []
    for parte in indice["partes"]:
        mods = [c for c, m in zip(claves, deck.MODULOS) if m.get("paso")] if parte["mods"] == "pasos" else parte["mods"]
        cubiertos += mods
        assert len(parte["tramos"]) == len(mods) and all(x > 0 for x in parte["tramos"])
        assert parte["min"] == max(1, round(sum(parte["tramos"])))
    assert cubiertos == claves[1:], "el índice debe seguir el orden de los capítulos y no saltarse ninguno"
    minutos = deck.minutos_por_modulo(laminas)
    assert abs(sum(sum(p["tramos"]) for p in indice["partes"]) - sum(v for k, v in minutos.items() if k)) < 0.2
    ctx = dict(kicker=indice["kicker"], modulo=0, n_modulos=indice["n_modulos"], avance=indice["avance"])
    html = pagina(indice, ctx)
    assert html.count('class="ix"') == len(indice["partes"])
    assert html.count('<i style="flex:') == len(claves) - 1          # un tramo de la barra por capítulo


def test_primeros_pasos_explica_la_descarga_y_los_planes():
    """Al comienzo, un capítulo dice dónde se usa Claude, cómo se instala y qué planes hay."""
    laminas = _laminas()
    assert deck.MODULOS[1]["clave"] == "m01_primeros"
    cap = [s for s in laminas if s["modulo"] == 1]
    assert cap[0]["tipo"] == "seccion" and {"accesos", "instalar", "planes"} <= {s["tipo"] for s in cap}
    html = ""
    for s in cap:
        html += pagina(s, dict(kicker=s["kicker"], modulo=s["modulo"], n_modulos=s["n_modulos"], avance=s["avance"]))
    for esperado in ("claude.ai", "claude.com/download", "claude.com/pricing", "iPhone y Android", "Free", "Pro", "Max",
                     "Ultra", "Team", "Enterprise", "Education", "sin impuestos"):
        assert esperado in html, esperado
    voz = " ".join(g["voz"] for g in deck.segmentos(laminas) if set(g["laminas"]) & {s["id"] for s in cap})
    assert "$" not in voz and ".com" not in voz and "http" not in voz      # las direcciones y los precios se leen en pantalla
    for dicho in ("Fri", "Tim", "Énterprais", "aifon", "dólares"):
        assert dicho in voz, dicho


def test_modulos_en_orden_y_capitulos():
    laminas = _laminas()
    modulos = [s["modulo"] for s in laminas]
    assert modulos == sorted(modulos)
    assert set(modulos) == set(range(len(deck.MODULOS)))
    assert a_voz("Paso 3") == "Paso tres"


def test_la_marca_es_el_autor_con_su_linkedin():
    """El video se firma con el nombre del autor y el logo de LinkedIn; no nombra a ninguna empresa."""
    from curso_youtube.laminas import MARCA
    laminas = _laminas()
    for s in laminas:
        ctx = dict(kicker=s["kicker"], modulo=s["modulo"], n_modulos=s["n_modulos"], avance=s["avance"])
        html = pagina(s, ctx)
        assert "hyperion" not in html.lower(), s["id"]
        if s["tipo"] not in ("gancho", "titulo_curso", "chips", "seccion", "cierre") or 'class="pie"' in html:
            assert 'class="logo-in"' in html and MARCA in html, s["id"]
    for g in deck.segmentos(laminas):
        assert "hyperion" not in g["voz"].lower(), g["id"]


def test_el_curso_no_menciona_el_ebook():
    laminas = _laminas()
    for s in laminas:
        ctx = dict(kicker=s["kicker"], modulo=s["modulo"], n_modulos=s["n_modulos"], avance=s["avance"])
        assert "ebook" not in pagina(s, ctx).lower(), s["id"]
    for g in deck.segmentos(laminas):
        assert "ebook" not in g["texto"].lower() and "íbuk" not in g["voz"].lower(), g["id"]


def test_el_cierre_ofrece_asesoria_con_los_datos_de_contacto():
    """El video termina con el mensaje de asesoría, y los datos de contacto están en pantalla y en la descripción."""
    from curso_youtube import youtube
    from curso_youtube.laminas import CORREO, LINKEDIN_URL
    laminas = _laminas()
    cierre = laminas[-1]
    assert cierre["tipo"] == "cierre"
    ctx = dict(kicker=cierre["kicker"], modulo=cierre["modulo"], n_modulos=cierre["n_modulos"], avance=cierre["avance"])
    html = pagina(cierre, ctx)
    assert 'class="asesoria"' in html and CORREO in html and "linkedin.com/in/" in html
    voz = deck.segmentos(laminas)[-1]["voz"]
    assert "asesorarte" in voz and "@" not in voz and "linkedin.com" not in voz.lower()     # los datos se leen en pantalla
    descripcion = youtube.publicacion(["00:00 Introducción"], estimados=False)
    assert LINKEDIN_URL in descripcion and CORREO in descripcion
    assert "[ENLACE A TU PERFIL DE LINKEDIN]" not in descripcion


def test_la_ficha_no_deja_enlaces_por_completar():
    """La descripción y el comentario fijado llevan el enlace a los materiales: no queda nada entre corchetes."""
    import re
    from curso_youtube import youtube
    ficha = youtube.publicacion(["00:00 Introducción"], estimados=False)
    assert youtube.MATERIALES_URL.startswith("https://") and ficha.count(youtube.MATERIALES_URL) == 2
    assert not re.search(r"\[ENLACE[^\]]*\]", ficha)


def test_las_etiquetas_caben_en_el_maximo_de_youtube():
    from curso_youtube import youtube
    assert youtube.largo_de_etiquetas(youtube.ETIQUETAS) <= 500
    assert len(youtube.TITULOS[0]) <= 100


def test_los_prompts_para_descargar_salen_del_curso():
    """El documento que se comparte trae los 10 prompts, tal como se muestran en el video."""
    from curso_youtube import guion
    html = guion.prompts_html()
    numeros = {s["numero"] for s in _laminas() if s["tipo"] == "prompt" and "ejemplo" not in s["nombre"]}
    assert len(numeros) == 10 and all(f"PROMPT {n}</span>" in html for n in numeros)
    assert "Cinco repreguntas que siempre sirven" in html and "linkedin.com/in/" in html


def test_la_miniatura_usa_el_retrato_del_autor_y_archivos_que_existen():
    import re
    from pathlib import Path
    from curso_youtube import youtube
    raiz = Path(youtube.__file__).parent
    assert "__IMAGENES__/autor.png" in youtube.MINIATURA_HTML
    for carpeta, marca in (("imagenes", "__IMAGENES__"), ("fuentes", "__FUENTES__")):
        for archivo in re.findall(re.escape(marca) + r"/([\w.-]+)", youtube.MINIATURA_HTML):
            assert (raiz / carpeta / archivo).exists(), f"{carpeta}/{archivo}"
