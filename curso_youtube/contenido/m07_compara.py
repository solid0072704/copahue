"""Paso 6: desviaciones contra el presupuesto."""

from .. import caso, graficos
from ._util import con_signo, etiqueta, m, rank, var

D = caso.desviaciones()


def _fila(clave, etq=None, cls="", p=None, gasto=False):
    d = D[clave]
    favorable = (d["abs"] < 0) if gasto else (d["abs"] > 0)
    tono = "pos" if favorable else "neg"
    return dict(c=[etq or caso.ETIQUETAS[clave], m(d["ppto"]), m(d["real"]), (con_signo(d["abs"]), tono),
                   (var(d["pct"]), tono), etiqueta(favorable)], cls=cls, p=p)


LAMINAS = [
    dict(
        tipo="seccion", paso=6, etiqueta="Paso 6", nombre="Compara contra el presupuesto",
        bajada="¿Cumplimos lo que nos propusimos? ¿Dónde nos desviamos y por qué?",
        pasos=[
            "Paso 6: comparar contra el presupuesto. Hasta ahora comparamos contra el año anterior. Pero la "
            "pregunta de gestión es otra: ¿cumplimos lo que nos propusimos? Y si no, ¿dónde nos desviamos y por qué?",
        ],
    ),
    dict(
        tipo="puntos", titulo="Un buen análisis de desviaciones hace tres cosas", une=True,
        items=[
            dict(p=0, t="Mide", d="Cuánto se desvió cada línea, en pesos y en porcentaje"),
            dict(p=1, t="Prioriza", d="Fija un umbral de materialidad y se concentra en lo que lo supera"),
            dict(p=2, t="Explica", d="Separa el efecto del volumen del efecto del precio o del margen"),
        ],
        lateral=[dict(p=3, clase="oro", frase="No es lo mismo ganar menos porque vendiste menos, que ganar menos porque "
                                              "vendiste más barato.", ancho=600)],
        pasos=[
            "Un buen análisis de desviaciones hace tres cosas. Uno: mide. Cuánto se desvió cada línea, en pesos y "
            "en porcentaje.",
            "Dos: prioriza. No todas las desviaciones importan; hay que fijar un umbral de materialidad y "
            "concentrarse en lo que lo supera.",
            "Y tres: explica. Separa el efecto del volumen del efecto del precio o del margen.",
            "Porque no es lo mismo ganar menos porque vendiste menos, que ganar menos porque vendiste más barato. "
            "Las acciones son completamente distintas.",
        ],
    ),
    dict(
        tipo="prompt", titulo="Prompt 7: real contra presupuesto", numero=7, nombre="Análisis de desviaciones", alto=True,
        segs=[
            ("Compara el resultado real con el presupuesto del período.\n\n", None),
            ("1. Tabla: presupuesto, real, desviación en monto y en %.\n"
             "   Indica si cada una es favorable o desfavorable.\n", 1),
            ("2. Considera material toda desviación mayor a [5 %] y a\n"
             "   [20 millones]. Analiza solo esas.\n", 2),
            ("3. Descompón la desviación del margen bruto en efecto\n"
             "   volumen y efecto margen.\n", 3),
            ("4. Para cada desviación material, propón la causa más\n"
             "   probable y el dato que necesitas para confirmarla.\n", 4),
            ("5. Ordena de mayor a menor impacto en la utilidad.", 5),
        ],
        notas=[
            dict(p=1, t="Favorable o desfavorable", d="Evita confusiones con los signos"),
            dict(p=2, t="Materialidad", d="Ajusta los umbrales a tu empresa"),
            dict(p=3, t="Volumen y margen", d="¿Vendí menos, o vendí más barato?"),
            dict(p=4, t="Causa y dato para confirmar", d="Claude no sabe por qué; sí qué revisar"),
        ],
        pasos=[
            "Vamos al prompt.",
            "Primero, la tabla clásica, indicando si cada desviación es favorable o desfavorable. Eso evita "
            "confusiones con los signos: gastar más que el presupuesto es desfavorable, vender más es favorable.",
            "Segundo, la materialidad: le digo que analice solo las desviaciones mayores a 5 % y a 20 millones. "
            "Ajusta esos umbrales a tu empresa.",
            "Tercero, que descomponga la desviación del margen bruto en efecto volumen y efecto margen.",
            "Cuarto, y esto es clave: para cada desviación, la causa más probable y el dato que necesita para "
            "confirmarla. Claude no sabe por qué subieron los fletes. Pero puede decirte qué revisar.",
            "Y quinto, que ordene por impacto en la utilidad.",
        ],
    ),
    dict(
        tipo="tabla", titulo="Alerce contra su presupuesto: vendió más y ganó menos", compacta=True, flex=2.5, alto=True,
        columnas=["Millones de pesos", "Ppto.", "Real", "Desv.", "%", ""],
        filas=[
            _fila("ventas", p=0),
            _fila("margen_bruto", cls="total", p=1),
            _fila("gav", etq="Gastos de adm. y ventas", p=2, gasto=True),
            _fila("distribucion", etq="Distribución y fletes", cls="sub", p=2, gasto=True),
            _fila("ebit", cls="total"),
            _fila("gastos_fin", p=3, gasto=True),
            _fila("utilidad", cls="total", p=4),
        ],
        notas=[
            dict(p=1, gr="+600 → +38", t="Más venta, casi el mismo margen"),
            dict(p=2, gr="+19 %", tono="neg", t="Fletes sobre el presupuesto"),
            dict(p=3, gr="+46 %", tono="neg", t="Gastos financieros"),
            dict(p=4, gr="−82", tono="neg", t="Utilidad bajo el presupuesto", clase="rojo"),
        ],
        pasos=[
            "Veamos Alerce contra su presupuesto. Las ventas superaron el presupuesto en 600 millones, un 5,4 %. "
            "Hasta ahí, una buena noticia.",
            "Pero el margen bruto prácticamente no se movió: solo 38 millones más. Vendió 600 millones adicionales "
            "y casi no ganó nada con ellos.",
            "Los gastos se pasaron en 90 millones, y toda la desviación viene de una línea: distribución y fletes, "
            "19 % sobre el presupuesto.",
            "Los gastos financieros se pasaron en 55 millones, 46 % más de lo previsto.",
            "Y el resultado final: la utilidad quedó 82 millones bajo el presupuesto, un 14,5 % menos. La empresa "
            "vendió más de lo que planificó, y ganó menos de lo que planificó.",
        ],
    ),
    dict(
        tipo="svg", titulo="600 millones más de venta, solo 38 más de margen",
        svg=graficos.puente_margen(),
        notas=[
            dict(p=1, gr="+156", tono="pos", t="Efecto volumen", d="600 millones más, al 26 % presupuestado"),
            dict(p=2, gr="−118", tono="neg", t="Efecto margen", d="1 punto menos sobre 11.800 millones"),
            dict(p=3, gr="¾", t="del beneficio de vender más", d="se fue en vender más barato"),
        ],
        pasos=[
            "¿Cómo se explica que 600 millones más de venta dejen solo 38 de margen? Con la descomposición.",
            "El efecto volumen: si esos 600 millones adicionales se hubieran vendido con el margen presupuestado, "
            "de 26 %, habrían aportado 156 millones.",
            "El efecto margen: como toda la venta se hizo con un punto menos de margen, se perdieron 118 millones.",
            "156 menos 118: los 38 que vimos. Dicho de otra forma, tres cuartas partes del beneficio de vender más "
            "se fueron en vender más barato.",
            "Este cálculo, que a mano toma un buen rato y en el que es fácil equivocarse, Claude lo entrega en "
            "segundos y con la fórmula a la vista.",
        ],
    ),
    dict(
        tipo="html", titulo="Las dos fórmulas, por si quieres revisarlas",
        html=('<div class="columna" style="gap:26px">'
              '<div class="tarjeta" data-p="0"><div class="eti">Efecto volumen</div>'
              '<div class="frase">(Ventas reales − ventas presupuestadas) × margen presupuestado</div>'
              f'<p>(11.800 − 11.200) × 26 % = <b class="pos">+{m(round(D["efecto_volumen"]))}</b></p></div>'
              '<div class="tarjeta" data-p="1"><div class="eti">Efecto margen</div>'
              '<div class="frase">(Margen real − margen presupuestado) × ventas reales</div>'
              f'<p>(25 % − 26 %) × 11.800 = <b class="neg">{m(round(D["efecto_margen"]))}</b></p></div>'
              '<div class="tarjeta oro" data-p="2"><p><b>Comprobación:</b> +156 − 118 = +38, exactamente la desviación del margen bruto.</p></div>'
              '</div>'),
        pasos=[
            "Las dos fórmulas, por si quieres revisarlas. Efecto volumen: la diferencia entre las ventas reales y "
            "las presupuestadas, multiplicada por el margen presupuestado.",
            "Efecto margen: la diferencia entre el margen real y el presupuestado, multiplicada por las ventas "
            "reales.",
            "Si sumas los dos efectos, obtienes exactamente la desviación del margen bruto. Esa es tu comprobación.",
        ],
    ),
    dict(
        tipo="html", titulo="Dónde poner el foco",
        html=rank([
            dict(p=0, n=1, t="Margen", d="Causa probable: descuentos a las cadenas · Confirmar con: margen por cliente",
                 v="−118", unidad="millones"),
            dict(p=1, n=2, t="Distribución y fletes", d="Causa probable: más despachos y más lejanos · Confirmar con: costo por ruta",
                 v="−90", unidad="millones"),
            dict(p=2, n=3, t="Gastos financieros", d="Causa probable: uso de la línea de crédito · Confirmar con: saldo promedio y tasa",
                 v="−55", unidad="millones"),
        ]),
        pie=dict(p=3, t="Tres desviaciones, un mismo origen: la forma en que la empresa creció."),
        pasos=[
            "Y este es el foco que propone, ordenado por impacto. Primero, el margen: 118 millones. Causa probable, "
            "los descuentos a las cadenas. Dato para confirmar, el margen por cliente, que ya vimos.",
            "Segundo, los fletes: 90 millones. Causa probable, más despachos y a mayor distancia. Dato para "
            "confirmar, el costo por despacho y por ruta.",
            "Tercero, los gastos financieros: 55 millones, por el uso de la línea de crédito para financiar el "
            "capital de trabajo.",
            "Fíjate que las tres desviaciones tienen un origen común: la forma en que la empresa creció. Y eso es "
            "lo que un buen análisis de desviaciones debería entregarte. No una tabla con 30 líneas en rojo y verde, "
            "sino dos o tres focos con nombre y apellido.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Tres consejos para este paso",
        items=[
            dict(p=0, icono="calendario", t="Compara mes a mes y también en acumulado",
                 d="Una desviación pequeña que se repite termina siendo grande"),
            dict(p=1, icono="sube", t="Pide una proyección de cierre", d="Con lo que va del año, ¿cómo terminamos?"),
            dict(p=2, icono="carpeta", t="Guarda las explicaciones en tu proyecto", d="Las causas de este mes son el contexto del próximo"),
        ],
        pasos=[
            "Tres consejos para sacarle más partido a este paso. Uno: compara mes a mes, y también en acumulado. Una "
            "desviación pequeña que se repite todos los meses termina siendo una desviación grande a fin de año, y "
            "el acumulado la muestra antes.",
            "Dos: pídele una proyección de cierre. Con lo que va del año, y con lo que queda del presupuesto, ¿cómo "
            "terminaríamos? Es la pregunta que más le interesa a un directorio a mitad de año.",
            "Y tres: guarda las explicaciones. Las causas que identificaste este mes son el contexto del mes "
            "siguiente. Si las dejas en tu proyecto, Claude las va a tener presentes en el próximo cierre.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Paso 6 en resumen", estilo="check",
        items=[
            dict(p=0, t="Mide, prioriza y explica", d="Con un umbral de materialidad"),
            dict(p=1, t="Separa el efecto volumen del efecto margen", d="¿Vendiste menos, o vendiste más barato?"),
            dict(p=2, t="Termina con dos o tres focos", d="No con 30 líneas en rojo y verde"),
        ],
        pasos=[
            "El paso 6, en resumen. Mide, prioriza y explica.",
            "Separa el efecto volumen del efecto margen.",
            "Y termina con dos o tres focos, no con 30 líneas en rojo y verde.",
        ],
    ),
]
