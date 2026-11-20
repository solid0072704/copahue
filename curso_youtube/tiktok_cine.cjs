// Reproduce una lámina del curso con otro ritmo y guarda, cuadro a cuadro, solo la tarjeta que se verá en la ventana
// del video vertical (el prompt o la respuesta de Claude), ya recortada a la proporción de esa ventana.
//
//   node tiktok_cine.cjs <lamina.html> <carpeta_salida> <info.json>
//
// info.json: lo que recibe iniciarLamina (duraciones de cada paso, etc.) más cuadros (cuántos se guardan), fps,
// velocidad (cuánto avanza la lámina por cada segundo de video), desde (segundos de la lámina que se saltan al
// comienzo), selector (la tarjeta), margen (px alrededor de ella) y proporcion (ancho / alto de la ventana). Escribe 0001.jpg, 0002.jpg… y, por la salida estándar,
// {"n": cuadros, "recorte": {x, y, width, height}}.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const [html, salida, archivoInfo] = process.argv.slice(2);
  const info = JSON.parse(fs.readFileSync(archivoInfo, 'utf8'));
  fs.mkdirSync(salida, { recursive: true });
  const browser = await chromium.launch();
  const contexto = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const page = await contexto.newPage();
  const abrir = async () => {
    await page.goto('file://' + path.resolve(html));
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate((i) => { ajustar(); iniciarLamina(i); }, info);
  };
  const total = info.duraciones.reduce((a, b) => a + b, 0), n = info.cuadros, velocidad = info.velocidad || 1;
  // la tarjeta se mide en su estado final, cuando ya terminó de entrar; después se abre la lámina de nuevo,
  // porque el motor solo avanza: no vuelve limpio desde el final al primer cuadro
  await abrir();
  await page.evaluate((t) => cuadro(t), total);
  const caja = await page.evaluate((sel) => { const r = document.querySelector(sel).getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; }, info.selector);
  await abrir();
  let width = caja.w + 2 * info.margen, height = width / info.proporcion;
  if (height < caja.h + 2 * info.margen) { height = caja.h + 2 * info.margen; width = height * info.proporcion; }
  const recorte = { x: Math.round(caja.x + caja.w / 2 - width / 2), y: Math.round(caja.y + caja.h / 2 - height / 2), width: Math.round(width), height: Math.round(height) };
  for (let f = 0; f < n; f++) {
    await page.evaluate((t) => cuadro(t), Math.min(total, (info.desde || 0) + f * velocidad / info.fps));
    await page.screenshot({ path: path.join(salida, String(f + 1).padStart(4, '0') + '.jpg'), type: 'jpeg', quality: 95, clip: recorte });
  }
  await browser.close();
  console.log(JSON.stringify({ n, recorte }));
})().catch((e) => { console.error(e); process.exit(1); });
