// Captura las láminas a 1920 x 1080: la imagen final de cada paso y, si se pide, su animación cuadro a cuadro.
//
// Uso: node capturar.cjs <carpeta_html> <carpeta_frames> [--animar] [--tiempos pasos.json] [--hilos N]
//                        [--reanudar] [prefijo ...]
//
//   Sin --animar: una imagen por paso (<id>_pKK.png) y la revisión de desbordes. Sirve para revisar el diseño.
//   Con --animar: además, los cuadros de la animación de cada paso (<id>_pKK_fNNN.jpg), la imagen final también
//   en JPEG (<id>_pKK.jpg, la que usa el video) y animacion.json con cuántos cuadros tiene cada paso.
//   Con --tiempos (lo que dura cada paso en el video) la animación se ajusta para caber en su paso, y las
//   láminas con imagen de fondo se capturan completas, porque el fondo se mueve.
//   Con --reanudar no se repiten las láminas ya capturadas con el mismo HTML, estilo y tiempos.
const { chromium } = require('playwright');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const FPS = 30;
const CALIDAD = 95;
const dos = (n, largo = 2) => String(n).padStart(largo, '0');

(async () => {
  const args = process.argv.slice(2);
  const opcion = (nombre, porDefecto) => { const i = args.indexOf(nombre); if (i < 0) return porDefecto; const v = args[i + 1]; args.splice(i, 2); return v; };
  const bandera = (nombre) => { const i = args.indexOf(nombre); if (i < 0) return false; args.splice(i, 1); return true; };
  const animar = bandera('--animar');
  const reanudar = bandera('--reanudar');
  const archivoTiempos = opcion('--tiempos', null);
  const hilos = Math.max(1, parseInt(opcion('--hilos', animar ? '3' : '1'), 10));
  const [htmlDir, outDir, ...filtro] = args;
  const laminas = JSON.parse(fs.readFileSync(path.join(htmlDir, 'laminas.json'), 'utf8'))
    .filter((s) => !filtro.length || filtro.some((f) => s.id.startsWith(f)));
  // duración de cada paso en el video, por «id_pKK»
  const duraciones = archivoTiempos ? JSON.parse(fs.readFileSync(archivoTiempos, 'utf8')) : {};
  fs.mkdirSync(outDir, { recursive: true });

  // Todo lo que decide cómo se ve una lámina: su HTML, la hoja de estilos, el motor de movimiento y sus tiempos.
  const comunes = ['estilo.css', 'movimiento.js'].map((f) => fs.readFileSync(path.join(__dirname, f)));
  const firmaDe = (s) => {
    const h = crypto.createHash('sha1');
    h.update(fs.readFileSync(path.join(htmlDir, s.id + '.html')));
    comunes.forEach((c) => h.update(c));
    h.update(JSON.stringify([animar, FPS, CALIDAD, s.continua, Array.from({ length: s.pasos }, (_, k) => duraciones[`${s.id}_p${dos(k)}`])]));
    return h.digest('hex');
  };

  const cuadros = {};
  let avisos = 0, imagenes = 0, siguiente = 0, hechas = 0, saltadas = 0;

  async function trabajar() {
    // un navegador por hilo: dentro de un mismo navegador las capturas se hacen una tras otra
    const browser = await chromium.launch();
    const contexto = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
    const page = await contexto.newPage();
    for (;;) {
      const s = laminas[siguiente++];
      if (!s) break;
      const marca = path.join(outDir, s.id + '.listo'), firma = firmaDe(s);
      if (reanudar && fs.existsSync(marca)) {
        const previa = JSON.parse(fs.readFileSync(marca, 'utf8'));
        if (previa.firma === firma) { Object.assign(cuadros, previa.cuadros); saltadas++; continue; }
      }
      if (fs.existsSync(marca)) fs.unlinkSync(marca);
      const propios = {};
      await page.goto('file://' + path.resolve(htmlDir, s.id + '.html'));
      await page.evaluate(() => document.fonts.ready);
      await page.evaluate(() => { ajustar(); if (window.prepararLamina) prepararLamina(); });
      let tLamina = 0;
      for (let k = 0; k < s.pasos; k++) {
        const clave = `${s.id}_p${dos(k)}`;
        const durAnim = await page.evaluate((paso) => (window.prepararPaso ? prepararPaso(paso) : (irPaso(paso), 0)), k);
        const durPaso = duraciones[clave];                         // segundos que dura el paso en el video (si se conoce)
        if (animar) {
          // cuadros: la animación completa o, en láminas con fondo en movimiento, el paso entero
          let n = Math.ceil(durAnim * FPS) + 1, escala = 1;
          if (durPaso !== undefined) {
            const caben = Math.max(1, Math.round(durPaso * FPS) - 1);
            if (s.continua) n = caben;
            else if (n > caben) { escala = (n - 1) / Math.max(1, caben - 1); n = caben; }   // se acelera para caber
          }
          for (let f = 0; f < n; f++) {
            await page.evaluate(([paso, t, tl]) => irPaso(paso, t, tl), [k, (f / FPS) * escala, tLamina + f / FPS]);
            await page.screenshot({ path: path.join(outDir, `${clave}_f${dos(f, 3)}.jpg`), type: 'jpeg', quality: CALIDAD });
          }
          propios[clave] = n;
          imagenes += n;
          if (durPaso !== undefined) tLamina += durPaso;
        }
        // imagen final del paso; en láminas con fondo en movimiento, tal como queda al terminar el paso
        const tFondo = s.continua && durPaso !== undefined ? tLamina : 0;
        await page.evaluate(([paso, tl]) => (window.prepararPaso ? irPaso(paso, null, tl) : irPaso(paso)), [k, tFondo]);
        await page.screenshot({ path: path.join(outDir, `${clave}.png`) });
        if (animar) await page.screenshot({ path: path.join(outDir, `${clave}.jpg`), type: 'jpeg', quality: CALIDAD });
        imagenes++;
      }
      const fuera = await page.evaluate(() => revisar());
      if (fuera.length) { avisos++; console.log('AVISO', s.id, fuera.join(' | ')); }
      Object.assign(cuadros, propios);
      if (animar) fs.writeFileSync(marca, JSON.stringify({ firma, cuadros: propios }));
      hechas++;
      if (animar) console.log(`${s.id} listo (${hechas + saltadas}/${laminas.length})`);
    }
    await browser.close();
  }
  await Promise.all(Array.from({ length: Math.min(hilos, laminas.length) }, trabajar));
  if (animar) {
    const destino = path.join(outDir, 'animacion.json');
    const previo = fs.existsSync(destino) && filtro.length ? JSON.parse(fs.readFileSync(destino, 'utf8')) : {};
    fs.writeFileSync(destino, JSON.stringify(Object.assign(previo, cuadros)));
  }
  console.log(`${imagenes} imágenes; ${avisos} láminas con avisos` + (saltadas ? `; ${saltadas} láminas ya estaban capturadas` : ''));
})().catch((e) => { console.error(e); process.exit(1); });
