/**
 * Vista previa de TPDV Academy: ejecuta el servidor real sobre el simulador de Apps Script,
 * abre la página en Chromium (Playwright) y guarda capturas de cada pantalla.
 *
 * Uso:
 *   node academia/tests/vista_previa.js --salida <carpeta> [--logo <png>] [--fuentes <node_modules/@fontsource>]
 *
 * Los visores de Drive se reemplazan por un marcador (no hay conexión a Google desde aquí).
 */
'use strict';

const fs = require('fs');
const path = require('path');
const { chromium, devices } = require('playwright');
const { ADMIN, entornoInstalado, respuestasCorrectas } = require('./fixtures');

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, v, i, arr) => {
  if (v.startsWith('--')) acc.push([v.slice(2), arr[i + 1]]);
  return acc;
}, []));
const SALIDA = path.resolve(args.salida || 'vista_previa');
fs.mkdirSync(SALIDA, { recursive: true });

const LOGO_BLANCO = '1XWLUOPw-nXc8ps8VOdk_sc7glwxIk_dD';
const LOGO_COLOR = '1PpMp46Vcb0dZiZeGX9L1aglMz1APL_RM';

const env = entornoInstalado((arbol) => {
  if (args.logo) arbol.archivos[LOGO_BLANCO].bytes = [...fs.readFileSync(args.logo)];
  else delete arbol.archivos[LOGO_BLANCO];
  delete arbol.archivos[LOGO_COLOR]; // sin el PNG multicolor, el certificado usa el logotipo tipográfico
});

/* ------------------------------------------------------------ datos de ejemplo */

function avanzar(email, nombre, area, hasta) {
  env.comoUsuario(email);
  env.ejecutar('doGet');
  env.ejecutar('guardarPerfil', { nombre, area });
  hasta(email);
}
function completarLecciones(idModulo, cantidad) {
  const modulo = env.ejecutar('obtenerDatos').programa.modulos.find(m => m.id === idModulo);
  modulo.lecciones.slice(0, cantidad === undefined ? modulo.lecciones.length : cantidad).forEach(l => env.ejecutar('marcarLeccion', l.id, true));
}
function aprobar(idModulo) {
  completarLecciones(idModulo);
  env.ejecutar('enviarEvaluacion', idModulo, respuestasCorrectas(env, idModulo));
}

const CAMILA = 'camila.rojas@tuproyectodevida.cl';
const DIEGO = 'diego.soto@tuproyectodevida.cl';
const VALENTINA = 'valentina.munoz@tuproyectodevida.cl';
const NUEVO = 'tomas.fuentes@tuproyectodevida.cl';

avanzar(CAMILA, 'Camila Rojas', 'Ventas zona sur', () => { aprobar('INDUCCION'); completarLecciones('VENTAS-1', 5); });
avanzar(DIEGO, 'Diego Soto', 'Ventas zona centro', () => {
  aprobar('INDUCCION');
  completarLecciones('VENTAS-1');
  env.ejecutar('enviarEvaluacion', 'VENTAS-1', { 1: 'B', 2: 'C', 3: 'A', 4: 'A', 5: 'B', 6: 'D' });
});
avanzar(VALENTINA, 'Valentina Muñoz', 'Ventas zona sur', () => {
  ['INDUCCION', 'VENTAS-1', 'VENTAS-2', 'MARKETING-1', 'MARKETING-2', 'LEGAL-1', 'LEGAL-2'].forEach(aprobar);
  completarLecciones('FINAL');
});
env.comoUsuario(ADMIN);
env.ejecutar('registrarEvaluacionPractica', VALENTINA, 'FINAL', 90, 'Excelente presentación del proyecto y manejo de objeciones. Reforzar plazos legales.');
env.ejecutar('doGet');
env.ejecutar('guardarPerfil', { nombre: 'Coordinación Academia', area: 'Gerencia comercial' });

/* ------------------------------------------------------------ navegador */

const MARCADOR_DRIVE = (url) => `<!doctype html><html><body style="margin:0;font-family:Arial,sans-serif;background:#fff;display:grid;place-items:center;height:100vh;color:#5B6873">
<div style="text-align:center;max-width:420px;padding:24px"><div style="width:64px;height:80px;margin:0 auto 16px;border:2px solid #DDE4EA;border-radius:8px;background:linear-gradient(#F0F8FF 0 18%,#fff 18%)"></div>
<strong style="color:#1A1A1A">Vista previa de Google Drive</strong><br>Aquí se muestra el archivo real cuando la academia corre en Apps Script.<br>
<small style="word-break:break-all;color:#9AA7B2">${url.replace(/[<>&"]/g, '')}</small></div></body></html>`;

function cssFuentes() {
  if (!args.fuentes) return '';
  const caras = [['Montserrat', 'montserrat', [400, 500, 600, 700]], ['Lato', 'lato', [400, 700]]];
  return caras.map(([familia, carpeta, pesos]) => pesos.map(p =>
    `@font-face{font-family:'${familia}';font-style:normal;font-weight:${p};font-display:swap;src:url(https://fuentes.local/${carpeta}/files/${carpeta}-latin-${p}-normal.woff2) format('woff2');}`).join('\n')).join('\n');
}

async function prepararPagina(navegador, email, opciones = {}) {
  const contexto = await navegador.newContext(opciones.dispositivo || { viewport: { width: 1366, height: 900 }, deviceScaleFactor: 2, locale: 'es-CL' });
  const pagina = await contexto.newPage();
  const errores = [];
  pagina.on('pageerror', (e) => errores.push(e.message));
  pagina.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errores.push(m.text()); });

  env.comoUsuario(email);
  const html = env.ejecutar('doGet').paginaCompleta();
  await pagina.route('**/*', (ruta) => {
    const url = ruta.request().url();
    if (url === 'https://academia.local/') return ruta.fulfill({ contentType: 'text/html; charset=utf-8', body: html });
    if (url.startsWith('https://fonts.googleapis.com/')) return ruta.fulfill({ contentType: 'text/css', body: cssFuentes() });
    if (url.startsWith('https://fuentes.local/') && args.fuentes) {
      const archivo = path.join(args.fuentes, url.replace('https://fuentes.local/', ''));
      return fs.existsSync(archivo) ? ruta.fulfill({ contentType: 'font/woff2', body: fs.readFileSync(archivo) }) : ruta.abort();
    }
    if (/^https:\/\/(drive|docs)\.google\.com\//.test(url)) return ruta.fulfill({ contentType: 'text/html; charset=utf-8', body: MARCADOR_DRIVE(url) });
    if (url.startsWith('about:') || url.startsWith('data:')) return ruta.continue();
    return ruta.abort();
  });

  await pagina.exposeFunction('__llamar', (nombre, parametros) => {
    env.comoUsuario(email);
    try { return { ok: env.ejecutar(nombre, ...parametros) }; } catch (e) { return { error: e.message }; }
  });
  await pagina.addInitScript(() => {
    const crear = (exito, fallo) => new Proxy({}, {
      get(_, prop) {
        if (prop === 'withSuccessHandler') return (f) => crear(f, fallo);
        if (prop === 'withFailureHandler') return (f) => crear(exito, f);
        return (...a) => window.__llamar(prop, a).then(r => (r.error ? fallo && fallo(new Error(r.error)) : exito && exito(r.ok)));
      }
    });
    window.google = { script: { run: crear(null, null), history: { push() {}, setChangeHandler() {} } } };
  });

  await pagina.goto('https://academia.local/', { waitUntil: 'load' });
  await pagina.evaluate(() => document.fonts.ready);
  return { pagina, contexto, errores };
}

async function captura(pagina, nombre, completa = true) {
  await pagina.waitForTimeout(350);
  await pagina.evaluate(() => document.querySelectorAll('.aviso').forEach(a => a.remove()));
  await pagina.screenshot({ path: path.join(SALIDA, nombre), fullPage: completa });
  console.log('✓', nombre);
}

(async () => {
  const navegador = await chromium.launch();
  const todosLosErrores = [];
  try {
    // 0. Primer ingreso: se pide el nombre para el certificado.
    let s = await prepararPagina(navegador, NUEVO);
    await captura(s.pagina, '00_primer_ingreso.png', false);
    await s.pagina.fill('#pf-nombre', 'Tomás Fuentes');
    await s.pagina.click('#form-perfil button[type=submit]');
    await s.pagina.waitForSelector('.modal', { state: 'hidden' });
    todosLosErrores.push(...s.errores);
    await s.contexto.close();

    // 1. Ruta de una asesora en su primera semana.
    s = await prepararPagina(navegador, CAMILA);
    await captura(s.pagina, '01_mi_ruta.png');
    await s.pagina.click('[data-ir="modulo"][data-modulo="VENTAS-1"]');
    await captura(s.pagina, '02_modulo.png', false);
    await s.pagina.click('[data-ir="leccion"][data-leccion="V1-06"]');
    await captura(s.pagina, '03_leccion_drive.png', false);
    await s.pagina.click('#btn-completar');
    await s.pagina.waitForSelector('#btn-completar.boton--exito');
    await s.pagina.click('.cabecera__nav [data-ir="inicio"]');
    await s.pagina.click('[data-ir="modulo"][data-modulo="INDUCCION"]');
    await s.pagina.click('[data-ir="leccion"][data-leccion="IND-03"]');
    await captura(s.pagina, '04_lectura.png');
    todosLosErrores.push(...s.errores);
    await s.contexto.close();

    // 2. Evaluación del módulo y resultado.
    s = await prepararPagina(navegador, DIEGO);
    await s.pagina.click('[data-ir="modulo"][data-modulo="VENTAS-1"]');
    await s.pagina.click('[data-ir="evaluacion"][data-modulo="VENTAS-1"]');
    await s.pagina.waitForSelector('#form-evaluacion');
    const correctas = respuestasCorrectas(env, 'VENTAS-1');
    for (const [n, letra] of Object.entries(correctas)) {
      await s.pagina.check(`input[name="p${n}"][value="${n === '6' ? 'A' : letra}"]`);
    }
    await captura(s.pagina, '05_evaluacion.png', false);
    await s.pagina.click('#btn-enviar');
    await s.pagina.waitForSelector('.resultado');
    await captura(s.pagina, '06_resultado.png');
    todosLosErrores.push(...s.errores);
    await s.contexto.close();

    // 3. Certificado de quien completó el programa.
    s = await prepararPagina(navegador, VALENTINA);
    await captura(s.pagina, '07_ruta_completa.png', false);
    await s.pagina.click('.cabecera__nav [data-ir="certificado"]');
    await captura(s.pagina, '08_certificado.png');
    todosLosErrores.push(...s.errores);
    await s.contexto.close();

    // 4. Panel de administración.
    s = await prepararPagina(navegador, ADMIN);
    await s.pagina.click('.cabecera__nav [data-ir="panel"]');
    await s.pagina.waitForSelector('.kpis');
    await s.pagina.click('#btn-permisos');
    await s.pagina.waitForSelector('.problema, .kpis + section');
    await captura(s.pagina, '09_panel_admin.png');
    await s.pagina.click('.btn-practica-fila[data-email="' + DIEGO + '"]');
    await s.pagina.fill('#pr-puntos', '25');
    await captura(s.pagina, '10_registrar_practica.png', false);
    todosLosErrores.push(...s.errores);
    await s.contexto.close();

    // 5. Celular.
    s = await prepararPagina(navegador, CAMILA, { dispositivo: Object.assign({}, devices['iPhone 13'], { locale: 'es-CL' }) });
    await captura(s.pagina, '11_movil_ruta.png', false);
    await s.pagina.click('[data-ir="continuar"]');
    await captura(s.pagina, '12_movil_leccion.png', false);
    const desborde = await s.pagina.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    if (desborde > 0) todosLosErrores.push('Desborde horizontal en móvil: ' + desborde + 'px');
    todosLosErrores.push(...s.errores);
    await s.contexto.close();

    // 6. Cuenta externa.
    s = await prepararPagina(navegador, 'persona.externa@gmail.com');
    await captura(s.pagina, '13_acceso_restringido.png', false);
    todosLosErrores.push(...s.errores);
    await s.contexto.close();
  } finally {
    await navegador.close();
  }
  if (todosLosErrores.length) {
    console.error('Errores en la página:\n' + todosLosErrores.join('\n'));
    process.exit(1);
  }
  console.log('Capturas en ' + SALIDA);
})();
