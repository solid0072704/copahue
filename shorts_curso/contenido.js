/* Contenido de los 4 Shorts del curso «Análisis de estados financieros con Claude».
   Cada escena: { d: duración en segundos, html }.
   data-in = segundo (dentro de la escena) en que entra el elemento · data-fx = animación.
   Todas las cifras salen del caso Distribuidora Alerce (empresa ficticia, millones de pesos). */

// Cambia aquí el llamado a la acción del cierre de los 4 videos.
const CTA = 'Curso completo en el canal';
const AUTOR = 'Rodrigo Barceló';

const cierre = (sub) => ({
  d: 4.6,
  html: `
    <div class="k" data-in=".1">Curso</div>
    <div class="h2" data-in=".25">Análisis de estados financieros <span class="acc">con Claude</span></div>
    <div class="chips" data-in=".7"><span>8 pasos</span><span>10 prompts</span><span>1 caso completo</span></div>
    <div class="p" data-in="1.0">${sub}</div>
    <div class="cta" data-in="1.4" data-fx="pop">${CTA} →</div>
    <div class="by" data-in="1.8" data-fx="fade">por <b>${AUTOR}</b></div>`,
});

const prompt = (etiqueta, texto, dur) => `
  <div class="card" data-in=".3">
    <div class="top"><i></i><i></i><i></i><span>${etiqueta}</span></div>
    <div class="txt" data-in=".7" data-fx="type" data-dur="${dur}">${texto}</div>
  </div>`;

const PIE = '<div class="foot" data-in=".4" data-fx="fade">Caso ficticio del curso · cifras en millones de pesos</div>';

window.SHORTS = {
  /* ───────────────────────── 1 · El caso ───────────────────────── */
  1: {
    archivo: 'short-1-vendio-mas-y-se-quedo-sin-caja',
    titulo: 'Vendió 18 % más y se quedó sin caja',
    scenes: [
      { d: 4.0, cls: 'tight', html: `
        <div class="h1" data-in=".1">Vendió <span class="up">18&nbsp;% más.</span></div>
        <div class="h1" data-in="1.0">Ganó más.</div>
        <div class="h1" data-in="1.9" data-fx="pop" style="transform-origin:0 50%">Y se quedó <span class="dn">sin&nbsp;caja.</span></div>` },

      { d: 6.6, html: `
        <div class="k" data-in=".1">Caso · Distribuidora Alerce</div>
        <div class="h3" data-in=".2">2024 → 2025</div>
        <div class="stat" data-in=".5">
          <div class="lab">Ventas</div>
          <div class="chip up" data-in="1.4" data-fx="pop">▲ 18 %</div>
          <div class="val"><span class="from">10.000</span><span class="arr">→</span><b data-in=".7" data-fx="count" data-dur=".9" data-count="10000,11800"></b></div>
        </div>
        <div class="stat" data-in="1.9">
          <div class="lab">Utilidad neta</div>
          <div class="chip up" data-in="2.7" data-fx="pop">▲ 4 %</div>
          <div class="val"><span class="from">467</span><span class="arr">→</span><b data-in="2.1" data-fx="count" data-dur=".8" data-count="467,485"></b></div>
        </div>
        <div class="stat" data-in="3.3">
          <div class="lab">Caja</div>
          <div class="chip dn" data-in="4.4" data-fx="shake">▼ 90 %</div>
          <div class="val"><span class="from">620</span><span class="arr">→</span><b class="dn" data-in="3.5" data-fx="count" data-dur="1.1" data-count="620,65"></b></div>
        </div>
        ${PIE}` },

      { d: 3.0, html: `
        <div class="h1" style="font-size:150px" data-in=".1">¿Dónde quedó</div>
        <div class="h1" style="font-size:150px" data-in=".45">la <span class="hl">plata?<u data-in=".9" data-fx="grow" data-dur=".5"></u></span></div>` },

      { d: 7.8, cls: 'tight', html: `
        <div class="k" data-in=".1">Atrapada en el capital de trabajo</div>
        <div class="blk" data-in=".4">
          <div class="lab">Cuentas por cobrar</div>
          <div class="big dn" data-in=".6" data-fx="count" data-dur=".9" data-count="0,960" data-sign="1"></div>
          <div class="bar"><b data-in=".6" data-fx="grow" data-dur="1" data-to="1"></b></div>
          <div class="sub"><span>Días de cobro:</span> 50 → 72</div>
        </div>
        <div class="blk" data-in="2.3">
          <div class="lab">Inventarios</div>
          <div class="big dn" data-in="2.5" data-fx="count" data-dur=".9" data-count="0,535" data-sign="1"></div>
          <div class="bar"><b data-in="2.5" data-fx="grow" data-dur="1" data-to=".56"></b></div>
          <div class="sub"><span>Días de inventario:</span> 55 → 68</div>
        </div>
        <div class="blk" data-in="4.3">
          <div class="lab">¿Y cómo se financió? Deuda de corto plazo</div>
          <div class="big">400 → <span class="dn" data-in="4.6" data-fx="count" data-dur=".9" data-count="400,1150"></span></div>
          <div class="sub"><span>Casi se triplicó.</span></div>
        </div>` },

      { d: 5.0, html: `
        <div class="h1" style="font-size:170px" data-in=".1">Utilidad</div>
        <div class="h1 acc" style="font-size:230px;line-height:.8" data-in=".6" data-fx="pop">≠</div>
        <div class="h1" style="font-size:170px" data-in="1.0">caja.</div>
        <div class="p" data-in="1.9">Flujo de la operación:<br><b>+510</b> → <b class="dn">−685</b></div>
        ${PIE}` },

      { d: 6.8, html: `
        <div class="k" data-in=".1">El prompt que lo destapa</div>
        ${prompt('Prompt 6 · Flujo de efectivo', 'Explícame qué pasó con la caja. (…) Responde en 5 líneas: ¿por qué la utilidad no se transformó en caja?', 3.4)}
        <div class="p" data-in="4.6">Pídele el <b>puente</b>: utilidad neta → flujo de la operación → variación del efectivo.</div>` },

      cierre('Este es el <b>Prompt 6 de 10</b>: sigue la caja.'),
    ],
  },

  /* ───────────────────────── 2 · Las 4 reglas ───────────────────────── */
  2: {
    archivo: 'short-2-4-reglas-antes-de-analizar-con-ia',
    titulo: '4 reglas antes de analizar tus estados financieros con IA',
    scenes: [
      { d: 4.4, cls: 'tight', html: `
        <div class="h2" data-in=".1">Antes de pasarle tus estados financieros a la IA…</div>
        <div class="h1" data-in="1.7" data-fx="pop" style="transform-origin:0 50%;margin-top:30px">dale estas <span class="acc">4&nbsp;reglas.</span></div>` },

      { d: 3.7, html: `
        <div class="num" data-in="0" data-fx="fade">1</div>
        <div class="k" data-in=".1">Regla 1 de 4</div>
        <div class="h2" data-in=".2">Usa <span class="acc">solo los datos</span> que te entrego.</div>
        <div class="p" data-in="1.1">Si falta algo, <b>pregunta</b>.</div>` },

      { d: 3.7, html: `
        <div class="num" data-in="0" data-fx="fade">2</div>
        <div class="k" data-in=".1">Regla 2 de 4</div>
        <div class="h2" data-in=".2">Calcula <span class="acc">con código</span></div>
        <div class="p" data-in="1.1">y <b>muestra la fórmula</b>.</div>` },

      { d: 3.7, html: `
        <div class="num" data-in="0" data-fx="fade">3</div>
        <div class="k" data-in=".1">Regla 3 de 4</div>
        <div class="h2" data-in=".2">Separa los <span class="acc">hechos</span> de tus <span class="acc">hipótesis.</span></div>` },

      { d: 3.9, html: `
        <div class="num" data-in="0" data-fx="fade">4</div>
        <div class="k" data-in=".1">Regla 4 de 4</div>
        <div class="h2" data-in=".2">No concluyas <span class="acc">todavía.</span></div>
        <div class="p" data-in="1.1">Primero <b>confirma qué entendiste</b>.</div>` },

      { d: 7.4, cls: 'tight', html: `
        <div class="k" data-in=".1">Y antes de analizar</div>
        <div class="h2" data-in=".2" style="margin-bottom:14px">Que todo <span class="acc">cuadre.</span></div>
        <div class="chk" data-in="1.0">
          <div class="eq">Activos = Pasivos + Patrimonio</div>
          <div class="nums">6.105 = 3.320 + 2.785 <span class="ok" data-in="1.7" data-fx="pop">✓</span></div>
        </div>
        <div class="chk" data-in="2.5">
          <div class="eq">Efectivo inicial + flujo neto = efectivo final</div>
          <div class="nums">620 − 555 = 65 <span class="ok" data-in="3.2" data-fx="pop">✓</span></div>
        </div>
        <div class="chk" data-in="4.0">
          <div class="eq">Utilidad = Δ util. acumuladas + dividendos</div>
          <div class="nums">485 = 285 + 200 <span class="ok" data-in="4.7" data-fx="pop">✓</span></div>
        </div>
        ${PIE}` },

      cierre('Son los <b>Prompts 1 y 2 de 10</b>: prepara y verifica.'),
    ],
  },

  /* ───────────────────────── 3 · El analista escéptico ───────────────────────── */
  3: {
    archivo: 'short-3-el-prompt-del-analista-esceptico',
    titulo: 'El prompt del analista escéptico',
    scenes: [
      { d: 4.6, cls: 'tight', html: `
        <div class="h2" data-in=".1">No le pidas a la IA que <span class="mut">analice</span> tu empresa.</div>
        <div class="h1" data-in="1.9" data-fx="pop" style="transform-origin:0 50%;margin-top:30px">Pídele que <span class="hl">dude<u data-in="2.5" data-fx="grow" data-dur=".5"></u></span> de&nbsp;ella.</div>` },

      { d: 6.8, html: `
        <div class="k" data-in=".1">Copia este prompt</div>
        ${prompt('Prompt 8 · Señales de alerta', 'Actúa como un analista de riesgo de crédito escéptico, que debe decidir si le presta dinero a esta empresa.', 3.6)}` },

      { d: 7.2, html: `
        <div class="k" data-in=".1">Y pídele 4 cosas</div>
        <div class="li" data-in=".4" data-fx="left"><i>1</i><div>Señales de alerta, con la <span class="acc">evidencia en cifras</span></div></div>
        <div class="li" data-in="1.7" data-fx="left"><i>2</i><div>¿La utilidad está <span class="acc">respaldada por caja</span>?</div></div>
        <div class="li" data-in="3.0" data-fx="left"><i>3</i><div>Supuestos <span class="acc">optimistas</span> de la administración</div></div>
        <div class="li" data-in="4.3" data-fx="left"><i>4</i><div>Clasificar: crítica, importante o a observar</div></div>` },

      { d: 8.2, html: `
        <div class="k" data-in=".1">Lo que aparece en el caso del curso</div>
        <div class="flag" data-in=".5" data-fx="left"><div class="lab">Utilidad vs. flujo de la operación</div><div class="big">+485 <span class="s">vs.</span> <span class="dn">−685</span></div><small>La utilidad no está respaldada por caja</small></div>
        <div class="flag" data-in="2.2" data-fx="left"><div class="lab">Cobranza vencida a más de 60 días</div><div class="big">40 <span class="s">→</span> <span class="dn">240</span></div><small>Y no hay provisión de incobrables</small></div>
        <div class="flag" data-in="3.9" data-fx="left"><div class="lab">Deuda bancaria de corto plazo</div><div class="big">400 <span class="s">→</span> <span class="dn">1.150</span></div><small>El crecimiento se financió con línea de crédito</small></div>
        ${PIE}` },

      { d: 4.2, cls: 'tight', html: `
        <div class="k" data-in=".1">Y cierra así</div>
        <div class="h1" data-in=".3">«Sé directo.</div>
        <div class="h1 acc" data-in="1.1">No suavices las conclusiones.»</div>` },

      cierre('Este es el <b>Prompt 8 de 10</b>: cuestiona.'),
    ],
  },

  /* ───────────────────────── 4 · Las 5 repreguntas ───────────────────────── */
  4: {
    archivo: 'short-4-5-repreguntas-que-siempre-sirven',
    titulo: '5 repreguntas que siempre sirven',
    scenes: [
      { d: 4.6, cls: 'tight', html: `
        <div class="h2" data-in=".1">La primera respuesta de la IA</div>
        <div class="h1 dn" data-in="1.0" data-fx="pop" style="transform-origin:0 50%">no es el análisis.</div>
        <div class="p" data-in="2.4" style="margin-top:26px">El análisis empieza cuando <b>repreguntas</b>.</div>` },

      { d: 2.8, cls: 'tight', html: `
        <div class="k" data-in=".1">Guárdalo</div>
        <div class="numsolid" style="font-size:330px" data-in=".15" data-fx="pop">5</div>
        <div class="h2" data-in=".5">repreguntas que siempre sirven</div>` },

      { d: 3.3, html: `
        <div class="num" data-in="0" data-fx="fade">1</div>
        <div class="k" data-in=".1">Repregunta 1</div>
        <div class="h1" data-in=".2">¿Por qué?</div>
        <div class="p" data-in="1.0">Dame <b>hipótesis</b> y el <b>dato</b> para confirmarlas.</div>` },

      { d: 3.3, html: `
        <div class="num" data-in="0" data-fx="fade">2</div>
        <div class="k" data-in=".1">Repregunta 2</div>
        <div class="h1" data-in=".2">¿Cómo lo<br>calculaste?</div>
        <div class="p" data-in="1.0">Muéstrame la <b>fórmula</b> y las <b>cifras</b>.</div>` },

      { d: 2.7, html: `
        <div class="num" data-in="0" data-fx="fade">3</div>
        <div class="k" data-in=".1">Repregunta 3</div>
        <div class="h1" data-in=".2">¿Qué<br>supusiste?</div>` },

      { d: 2.7, html: `
        <div class="num" data-in="0" data-fx="fade">4</div>
        <div class="k" data-in=".1">Repregunta 4</div>
        <div class="h1" data-in=".2">¿Qué dato<br>te falta?</div>` },

      { d: 2.9, html: `
        <div class="num" data-in="0" data-fx="fade">5</div>
        <div class="k" data-in=".1">Repregunta 5</div>
        <div class="h1" data-in=".2">¿Qué<br>pasaría si…?</div>` },

      { d: 7.0, html: `
        <div class="k" data-in=".1">Así se usa en el caso</div>
        ${prompt('Repregunta · Margen bruto', 'El margen bruto bajó 1 punto. Dame las 3 hipótesis más probables y, para cada una, el dato exacto que necesitas para confirmarla o descartarla.', 4.2)}` },

      cierre('Las repreguntas acompañan a los <b>10 prompts</b> del método.'),
    ],
  },
};
