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
