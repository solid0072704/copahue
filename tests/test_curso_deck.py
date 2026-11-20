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
    assert 55 <= minutos <= 63, f"duración estimada: {minutos:.1f} min"


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
