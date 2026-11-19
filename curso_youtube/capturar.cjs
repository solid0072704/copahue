// Captura cada lámina y cada paso a PNG de 1920 x 1080 y avisa si algo se sale del lienzo.
// Uso: node capturar.cjs <carpeta_html> <carpeta_frames> [prefijo ...]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const [htmlDir, outDir, ...filtro] = process.argv.slice(2);
  const laminas = JSON.parse(fs.readFileSync(path.join(htmlDir, 'laminas.json'), 'utf8'));
  const browser = await chromium.launch();
  const contexto = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const page = await contexto.newPage();
  let avisos = 0, n = 0;
  for (const s of laminas) {
    if (filtro.length && !filtro.some((f) => s.id.startsWith(f))) continue;
    await page.goto('file://' + path.resolve(htmlDir, s.id + '.html'));
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(() => ajustar());
    for (let k = 0; k < s.pasos; k++) {
      await page.evaluate((paso) => irPaso(paso), k);
      await page.screenshot({ path: path.join(outDir, `${s.id}_p${String(k).padStart(2, '0')}.png`) });
      n++;
    }
    const fuera = await page.evaluate(() => revisar());
    if (fuera.length) { avisos++; console.log('AVISO', s.id, fuera.join(' | ')); }
  }
  await browser.close();
  console.log(`${n} imágenes; ${avisos} láminas con avisos`);
})().catch((e) => { console.error(e); process.exit(1); });
