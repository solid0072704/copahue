// Captura las láminas (1920 x 1080) con Chromium: imágenes fijas para revisar el diseño o el video de cada escena.
//
//   node capturar.cjs <carpeta_html> <carpeta_frames> [--hilos N] [prefijo ...]
//       Imágenes fijas: una por paso (<id>_pKK.png) y la revisión de desbordes.
//
//   node capturar.cjs <carpeta_html> <carpeta_escenas> --continuo <plan.json> [--hilos N] [--reanudar] [prefijo ...]
//       Video: cada lámina se reproduce de corrido con movimiento.js (cuadro a cuadro, a los cuadros por segundo
//       del plan) y sus cuadros se pasan directo a ffmpeg, que deja <id>.mp4. No se guardan cuadros en disco.
//       El plan (lo escribe `construir --continuo`) dice cuántos cuadros dura cada lámina, cuánto dura cada paso y
//       cómo se codifica; con `ancho` y `alto` cambia el tamaño de la página (el video vertical usa 1080 x 1920).
//       Con --reanudar no se repiten las escenas ya hechas con el mismo HTML, estilo y plan.
//       De paso deja en <carpeta_escenas>/vistas una imagen por paso, tomada del video, para revisarlo.
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const CALIDAD = 93;             // de los cuadros JPEG que recibe ffmpeg
const dos = (n, largo = 2) => String(n).padStart(largo, '0');

(async () => {
  const args = process.argv.slice(2);
  const opcion = (nombre, porDefecto) => { const i = args.indexOf(nombre); if (i < 0) return porDefecto; const v = args[i + 1]; args.splice(i, 2); return v; };
  const bandera = (nombre) => { const i = args.indexOf(nombre); if (i < 0) return false; args.splice(i, 1); return true; };
  const archivoPlan = opcion('--continuo', null);
  const reanudar = bandera('--reanudar');
  const hilos = Math.max(1, parseInt(opcion('--hilos', archivoPlan ? '4' : '1'), 10));
  const [htmlDir, outDir, ...filtro] = args;
  const elegida = (s) => !filtro.length || filtro.some((f) => s.id.startsWith(f));
  fs.mkdirSync(outDir, { recursive: true });
  let avisos = 0, imagenes = 0, siguiente = 0, hechas = 0, saltadas = 0;

  // ---------------------------------------------------------------------------------------------- imágenes fijas
  async function fijas(laminas) {
    const browser = await chromium.launch();
    const contexto = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
    const page = await contexto.newPage();
    for (;;) {
      const s = laminas[siguiente++];
      if (!s) break;
      await page.goto('file://' + path.resolve(htmlDir, s.id + '.html'));
      await page.evaluate(() => document.fonts.ready);
      await page.evaluate(() => { ajustar(); if (window.prepararLamina) prepararLamina(); });
      for (let k = 0; k < s.pasos; k++) {
        await page.evaluate((paso) => (window.prepararPaso ? (prepararPaso(paso), irPaso(paso, null, 0)) : irPaso(paso)), k);
        await page.screenshot({ path: path.join(outDir, `${s.id}_p${dos(k)}.png`) });
        imagenes++;
      }
      const fuera = await page.evaluate(() => revisar());
      if (fuera.length) { avisos++; console.log('AVISO', s.id, fuera.join(' | ')); }
    }
    await browser.close();
  }

  // ------------------------------------------------------------------------------------------- video por escena
  let plan = null, comunes = [];
  // Todo lo que decide cómo se ve una escena: su HTML, sus imágenes, la hoja de estilos, el motor y su plan.
  const firmaDe = (e) => {
    const h = crypto.createHash('sha1');
    const html = fs.readFileSync(path.join(htmlDir, e.id + '.html'), 'utf8');
    h.update(html);
    comunes.forEach((c) => h.update(c));
    for (const m of html.matchAll(/imagenes\/([\w.-]+)/g)) h.update(fs.readFileSync(path.join(__dirname, 'imagenes', m[1])));
    h.update(JSON.stringify([e, plan.fps, plan.codificacion, CALIDAD]));
    return h.digest('hex');
  };
  async function escenas(lista) {
    const browser = await chromium.launch();     // un navegador por hilo: dentro de uno solo las capturas van de a una
    const contexto = await browser.newContext({ viewport: { width: plan.ancho || 1920, height: plan.alto || 1080 }, deviceScaleFactor: 1 });
    const page = await contexto.newPage();
    const cdp = await contexto.newCDPSession(page);      // la captura directa por el protocolo ahorra unos milisegundos por cuadro
    for (;;) {
      const e = lista[siguiente++];
      if (!e) break;
      const video = path.join(outDir, e.id + '.mp4'), marca = path.join(outDir, e.id + '.listo'), firma = firmaDe(e);
      if (reanudar && fs.existsSync(marca) && fs.existsSync(video) && JSON.parse(fs.readFileSync(marca, 'utf8')).firma === firma) { saltadas++; continue; }
      if (fs.existsSync(marca)) fs.unlinkSync(marca);
      await page.goto('file://' + path.resolve(htmlDir, e.id + '.html'));
      await page.evaluate(() => document.fonts.ready);
      await page.evaluate((info) => { ajustar(); iniciarLamina(info); }, e);
      // una vista por paso: el cuadro en que ya terminó de entrar todo, poco antes de pasar al siguiente
      const vistas = {};
      let t = 0;
      e.duraciones.forEach((d, k) => { vistas[Math.max(0, Math.min(e.cuadros - 1, Math.floor((t + Math.max(d * .5, d - .7)) * plan.fps)))] = `${e.id}_p${dos(k)}`; t += d; });
      const parcial = path.join(outDir, e.id + '.parcial.mp4');
      const ff = spawn('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-framerate', String(plan.fps), '-c:v', 'mjpeg', '-i', '-',
        ...plan.codificacion, '-frames:v', String(e.cuadros), '-an', parcial], { stdio: ['pipe', 'ignore', 'pipe'] });
      let error = '';
      ff.stderr.on('data', (d) => { error += d; });
      ff.stdin.on('error', () => {});
      const cierre = new Promise((ok, mal) => ff.on('close', (c) => (c === 0 ? ok() : mal(new Error(`ffmpeg (${e.id}): ${error.slice(-800)}`)))));
      for (let f = 0; f < e.cuadros; f++) {
        await page.evaluate((seg) => cuadro(seg), f / plan.fps);
        const jpg = Buffer.from((await cdp.send('Page.captureScreenshot', { format: 'jpeg', quality: CALIDAD, optimizeForSpeed: true })).data, 'base64');
        if (!ff.stdin.write(jpg)) await new Promise((r) => ff.stdin.once('drain', r));
        if (vistas[f]) fs.writeFileSync(path.join(outDir, 'vistas', vistas[f] + '.jpg'), jpg);
        imagenes++;
      }
      ff.stdin.end();
      await cierre;
      fs.renameSync(parcial, video);
      fs.writeFileSync(marca, JSON.stringify({ firma, cuadros: e.cuadros, cuadro0: e.cuadro0 }));
      hechas++;
      console.log(`${e.id} listo: ${e.cuadros} cuadros (${hechas + saltadas}/${lista.length})`);
    }
    await browser.close();
  }

  if (archivoPlan) {
    plan = JSON.parse(fs.readFileSync(archivoPlan, 'utf8'));
    comunes = ['estilo.css', 'movimiento.js'].map((f) => fs.readFileSync(path.join(__dirname, f)));
    fs.mkdirSync(path.join(outDir, 'vistas'), { recursive: true });
    const lista = plan.escenas.filter(elegida);
    await Promise.all(Array.from({ length: Math.min(hilos, lista.length) }, () => escenas(lista)));
    console.log(`${imagenes} cuadros en ${hechas} escenas` + (saltadas ? `; ${saltadas} escenas ya estaban hechas` : ''));
  } else {
    const laminas = JSON.parse(fs.readFileSync(path.join(htmlDir, 'laminas.json'), 'utf8')).filter(elegida);
    await Promise.all(Array.from({ length: Math.min(hilos, laminas.length) }, () => fijas(laminas)));
    console.log(`${imagenes} imágenes; ${avisos} láminas con avisos`);
  }
})().catch((e) => { console.error(e); process.exit(1); });
