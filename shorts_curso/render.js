#!/usr/bin/env node
/* Renderiza los Shorts a MP4 vertical (1080×1920, 30 fps).
   Uso:  node render.js            → los 4
         node render.js 2 4        → solo esos
         node render.js --stills   → una imagen por escena (revisión rápida) en salida/stills/
   Requiere: Node, Playwright (Chromium) y ffmpeg. */
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const FPS = 30;
const DIR = __dirname;
const OUT = path.join(DIR, 'salida');
const args = process.argv.slice(2);
const stills = args.includes('--stills');
const ids = args.filter((a) => /^\d+$/.test(a));

async function abrir(browser, id) {
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  await page.goto('file://' + path.join(DIR, 'short.html') + `?id=${id}&render=1`);
  await page.evaluate(() => document.fonts.ready);
  const meta = await page.evaluate(() => ({ total: window.TOTAL, scenes: window.SCENES, archivo: window.SHORTS[new URLSearchParams(location.search).get('id')].archivo }));
  return { page, ...meta };
}

async function video(browser, id) {
  const { page, total, archivo } = await abrir(browser, id);
  const destino = path.join(OUT, archivo + '.mp4');
  // voz en off: audio/short-N.m4a (ya sincronizado con tiempos.js); si no existe, pista en silencio
  const voz = path.join(DIR, 'audio', `short-${id}.m4a`);
  const audio = fs.existsSync(voz)
    ? ['-i', voz, '-af', 'apad', '-t', total.toFixed(3)]
    : ['-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', '-shortest'];
  const ff = spawn('ffmpeg', [
    '-y', '-loglevel', 'error',
    '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    ...audio,
    '-vf', 'scale=1080:1920:out_color_matrix=bt709:out_range=tv,format=yuv420p',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-r', String(FPS),
    '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
    '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-ac', '2', '-movflags', '+faststart', destino,
  ], { stdio: ['pipe', 'inherit', 'inherit'] });
  const fin = new Promise((ok, mal) => ff.on('close', (c) => (c === 0 ? ok() : mal(new Error('ffmpeg ' + c)))));
  const frames = Math.round(total * FPS);
  for (let f = 0; f < frames; f++) {
    await page.evaluate((t) => window.seek(t), f / FPS);
    const buf = await page.screenshot({ type: 'jpeg', quality: 96 });
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
  }
  ff.stdin.end();
  await fin;
  await page.close();
  console.log(`✓ Short ${id}: ${path.relative(DIR, destino)} (${total.toFixed(1)} s)`);
}

async function fotos(browser, id) {
  const { page, scenes } = await abrir(browser, id);
  const dir = path.join(OUT, 'stills');
  fs.mkdirSync(dir, { recursive: true });
  for (let i = 0; i < scenes.length; i++) {
    const [s, d] = scenes[i];
    await page.evaluate((t) => window.seek(t), s + d - 0.3);
    await page.screenshot({ path: path.join(dir, `s${id}-${String(i + 1).padStart(2, '0')}.png`) });
  }
  await page.close();
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const lista = ids.length ? ids : ['1', '2', '3', '4'];
  await Promise.all(lista.map((id) => (stills ? fotos(browser, id) : video(browser, id))));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
