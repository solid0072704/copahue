// Graba un video demo de Tu Cartera recorriendo la app real con el estudio de ejemplo.
//
//   TUCARTERA_HOY=2026-11-20 python -m tucartera --puerto 8765 &
//   node scripts/video_demo.js http://127.0.0.1:8765 salida/
//   ffmpeg -i salida/*.webm -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart tu-cartera-demo.mp4
//
// Necesita Playwright con Chromium. El video queda en la carpeta de salida como .webm.
const { chromium } = require("playwright");

const BASE = process.argv[2] || "http://127.0.0.1:8765";
const SALIDA = process.argv[3] || "video";
const W = 1440, H = 900;

// Cursor, subtítulos y portada que se dibujan sobre la app solo durante la grabación.
const CAPA = () => {
  if (window.top !== window) return;
  const instalar = () => {
    if (document.getElementById("__demo")) return;
    const capa = document.createElement("div");
    capa.id = "__demo";
    capa.innerHTML = `
      <style>
        #__cursor{position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;will-change:transform}
        #__onda{position:fixed;z-index:2147483646;pointer-events:none;width:44px;height:44px;margin:-22px 0 0 -22px;border-radius:50%;
          background:rgba(0,113,227,.28);transform:scale(.2);opacity:0}
        #__onda.activa{animation:__onda .5s ease-out}
        @keyframes __onda{0%{transform:scale(.2);opacity:1}100%{transform:scale(1.4);opacity:0}}
        #__rotulo{position:fixed;left:50%;bottom:30px;transform:translate(-50%,12px);z-index:2147483645;pointer-events:none;
          max-width:980px;padding:15px 26px;border-radius:18px;background:rgba(28,28,30,.9);color:#fff;text-align:center;
          font:600 20px/1.35 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif;letter-spacing:-.01em;
          box-shadow:0 14px 40px rgba(0,0,0,.25);opacity:0;transition:opacity .35s,transform .35s}
        #__rotulo.visible{opacity:1;transform:translate(-50%,0)}
        #__rotulo small{display:block;font-weight:500;font-size:15px;color:rgba(255,255,255,.7);margin-top:3px}
        #__portada{position:fixed;inset:0;z-index:2147483644;background:#f5f5f7;display:grid;place-items:center;text-align:center;
          transition:opacity .6s;font-family:-apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif;color:#1d1d1f}
        #__portada.oculta{opacity:0;pointer-events:none}
        #__portada h1{font-size:72px;letter-spacing:-.035em;margin:26px 0 10px}
        #__portada p{font-size:26px;color:#6e6e73;margin:0}
        .avisos{bottom:118px!important}
      </style>
      <div id="__portada" class="${location.search.includes("video=1") ? "" : "oculta"}"><div>
        <svg width="112" height="112" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#1d1d1f"/><g transform="translate(4 4)" fill="#fff">
        <rect x="3" y="3" width="8" height="8" rx="2.2"/><rect x="13" y="3" width="8" height="8" rx="2.2" opacity=".42"/>
        <rect x="3" y="13" width="8" height="8" rx="2.2" opacity=".42"/><rect x="13" y="13" width="8" height="8" rx="2.2"/></g></svg>
        <h1 id="__titulo">Tu Cartera</h1><p id="__bajada">Gestión de clientes para estudios contables</p></div></div>
      <div id="__rotulo"></div><div id="__onda"></div>
      <div id="__cursor"><svg width="24" height="30" viewBox="0 0 24 30"><path d="M3 2.5v21l5.6-5.2 3.7 8.4 3.6-1.6-3.6-8.2h7.6z"
        fill="#111" stroke="#fff" stroke-width="1.7" stroke-linejoin="round"/></svg></div>`;
    document.documentElement.appendChild(capa);
    window.__cursor(window.__x ?? 720, window.__y ?? 450);
    if (location.search.includes("video=1")) document.getElementById("__cursor").style.opacity = "0";
  };
  window.__cursor = (x, y) => {
    window.__x = x; window.__y = y;
    const c = document.getElementById("__cursor");
    if (c) c.style.transform = `translate(${x - 3}px, ${y - 2}px)`;
  };
  window.__clic = () => {
    const o = document.getElementById("__onda");
    o.style.left = `${window.__x}px`; o.style.top = `${window.__y}px`;
    o.classList.remove("activa"); void o.offsetWidth; o.classList.add("activa");
  };
  window.__rotulo = (texto, sub) => {
    const r = document.getElementById("__rotulo");
    if (!texto) { r.classList.remove("visible"); return; }
    r.innerHTML = texto + (sub ? `<small>${sub}</small>` : "");
    r.classList.add("visible");
  };
  window.__portada = (visible, titulo, bajada) => {
    const p = document.getElementById("__portada");
    if (titulo) document.getElementById("__titulo").textContent = titulo;
    if (bajada) document.getElementById("__bajada").textContent = bajada;
    p.classList.toggle("oculta", !visible);
    document.getElementById("__cursor").style.opacity = visible ? "0" : "1";
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", instalar);
  else instalar();
};

const VITRINA = `<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Tu Cartera en el celular</title></head>
<body style="margin:0;height:100vh;background:#f5f5f7;display:flex;align-items:center;justify-content:center;gap:110px;
font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif;color:#1d1d1f">
<div style="max-width:460px"><div style="font-size:15px;font-weight:600;color:#6e6e73;text-transform:uppercase;letter-spacing:.05em">Computador, tablet y celular</div>
<h1 style="font-size:56px;line-height:1.05;letter-spacing:-.035em;margin:.25em 0 .35em">También en tu celular.</h1>
<p style="font-size:22px;line-height:1.4;color:#6e6e73;margin:0">Se instala como app con su propio ícono. Revisa y aprueba desde cualquier lugar.</p></div>
<div style="width:404px;height:836px;border-radius:60px;background:#1d1d1f;padding:11px;box-sizing:border-box;box-shadow:0 30px 90px rgba(0,0,0,.25)">
<iframe src="/?video=0#/hoy" style="width:382px;height:814px;border:0;border-radius:50px;background:#f5f5f7;display:block"></iframe></div>
</body></html>`;

(async () => {
  const navegador = await chromium.launch();
  const contexto = await navegador.newContext({ viewport: { width: W, height: H }, recordVideo: { dir: SALIDA, size: { width: W, height: H } } });
  await contexto.addInitScript(CAPA);
  // La app no permite incrustarse en otros sitios; para la vitrina del celular se relaja solo aquí.
  await contexto.route(`${BASE}/**`, async (ruta) => {
    if (ruta.request().url().endsWith("/vitrina")) return ruta.fulfill({ contentType: "text/html", body: VITRINA });
    const r = await ruta.fetch();
    const encabezados = { ...r.headers() };
    delete encabezados["x-frame-options"];
    return ruta.fulfill({ response: r, headers: encabezados });
  });
  await contexto.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
  const p = await contexto.newPage();
  const espera = (ms) => p.waitForTimeout(ms);
  let pos = { x: 720, y: 450 };

  async function mover(x, y, pasos = 28) {
    const desde = { ...pos };
    for (let i = 1; i <= pasos; i++) {
      const t = i / pasos, e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      const xi = desde.x + (x - desde.x) * e, yi = desde.y + (y - desde.y) * e;
      await p.mouse.move(xi, yi);
      await p.evaluate(([a, b]) => window.__cursor(a, b), [xi, yi]);
      await espera(14);
    }
    pos = { x, y };
  }
  async function clic(localizador, { espera: ms = 650 } = {}) {
    const caja = await localizador.first().boundingBox();
    await mover(caja.x + caja.width / 2, caja.y + Math.min(caja.height / 2, 22));
    await espera(160);
    await p.evaluate(() => window.__clic());
    await p.mouse.click(pos.x, pos.y);
    await espera(ms);
  }
  const rotulo = (texto, sub = "") => p.evaluate(([t, s]) => window.__rotulo(t, s), [texto, sub]);
  const desplazar = (y) => p.evaluate((v) => window.scrollTo({ top: v, behavior: "smooth" }), y);

  // 1. Portada
  await p.goto(`${BASE}/?video=1`);
  await p.waitForSelector(".ingreso");
  await espera(3200);
  await p.evaluate(() => window.__portada(false));
  await espera(700);

  // 2. Ingreso
  await rotulo("Ingresa con Google, Microsoft o tu correo", "Cada estudio parte con 14 días de prueba gratis");
  const google = await p.locator(".proveedor", { hasText: "Google" }).boundingBox();
  await mover(google.x + google.width / 2 + 60, google.y + google.height / 2);
  await espera(1400);
  const microsoft = await p.locator(".proveedor", { hasText: "Microsoft" }).boundingBox();
  await mover(microsoft.x + microsoft.width / 2 + 60, microsoft.y + microsoft.height / 2, 16);
  await espera(1100);
  await clic(p.locator("[data-demo]"), { espera: 0 });
  await p.waitForSelector(".kpis", { timeout: 20000 });
  await espera(600);

  // 3. Hoy
  await rotulo("Hoy: lo urgente del día en un vistazo", "Propuestas por aprobar, agenda tributaria y clientes que requieren atención");
  await mover(560, 260);
  await espera(2200);
  await desplazar(420);
  await mover(1300, 520);
  await espera(2200);
  await desplazar(0);
  await espera(900);

  // 4. Para revisar
  await rotulo("Las automatizaciones dejan todo listo para aprobar", "Correos a clientes y tareas para el equipo, con el porqué de cada uno");
  await clic(p.locator(".nav a[data-ruta=revisar]"), { espera: 900 });
  await clic(p.locator("[data-filtro=correo]"), { espera: 500 });
  await clic(p.locator(".item", { hasText: "Pedir documentos" }), { espera: 1600 });
  await mover(900, 520);
  await espera(1800);
  await rotulo("Tú revisas, editas y apruebas", "Nada sale sin tu visto bueno · también con ⌘↵");
  await clic(p.locator("[data-aprobar]"), { espera: 2400 });

  // 5. Barra de pedidos
  await rotulo("Pide trabajo en lenguaje simple con ⌘K", "Busca clientes o encarga tareas a las automatizaciones");
  await p.keyboard.press("Control+k");
  await espera(700);
  await p.keyboard.type("revisa la cobranza de honorarios atrasados", { delay: 45 });
  await espera(500);
  await p.keyboard.press("Enter");
  await p.waitForSelector(".paleta-respuesta .icono-cuadro", { timeout: 15000 });
  await espera(2600);
  await p.keyboard.press("Escape");
  await espera(400);

  // 6. Clientes y ficha
  await rotulo("Toda tu cartera con su salud de 0 a 100", "Próximo vencimiento, documentos pendientes y deuda de cada cliente");
  await clic(p.locator(".nav a[data-ruta=clientes]"), { espera: 1200 });
  await mover(900, 520);
  await espera(1500);
  await clic(p.locator("tr", { hasText: "Ferretería El Roble" }), { espera: 900 });
  await rotulo("La ficha de cada cliente", "Ventas y compras de 12 meses, declaraciones, documentos y honorarios");
  await mover(640, 420);
  await espera(2600);

  // 7. Vencimientos
  await rotulo("Calendario tributario automático", "F29, Previred, LRE y F22 con días hábiles y feriados de Chile");
  await clic(p.locator(".nav a[data-ruta=vencimientos]"), { espera: 1200 });
  await mover(1060, 560);
  await espera(1600);
  await desplazar(560);
  await espera(2000);
  await desplazar(0);

  // 8. Documentos
  await rotulo("Documentos del mes, cliente por cliente", "Las solicitudes y recordatorios se redactan solos");
  await clic(p.locator(".nav a[data-ruta=documentos]"), { espera: 1200 });
  await clic(p.locator(".casilla[aria-checked=false]").nth(4), { espera: 1600 });

  // 9. Automatizaciones
  await rotulo("Seis automatizaciones trabajando por ti", "Tú decides si piden aprobación o actúan solas");
  await clic(p.locator(".nav a[data-ruta=automatizaciones]"), { espera: 1200 });
  await mover(1050, 420);
  await espera(2800);

  // 10. Plan y uso
  await rotulo("Suscripción por usuario con cuota mensual de IA", "Cada persona ve su consumo; si se agota, sigue trabajando en modo básico");
  await clic(p.locator(".nav a[data-ruta=plan]"), { espera: 1200 });
  await mover(800, 420);
  await espera(2200);
  await desplazar(760);
  await espera(2600);
  await desplazar(0);

  // 11. Ajustes: cuentas conectadas
  await rotulo("Conecta tu cuenta de Google o Microsoft", "Solo usamos tu nombre y correo para que ingreses");
  await clic(p.locator(".nav a[data-ruta=ajustes]"), { espera: 900 });
  await p.evaluate(() => document.querySelector("[data-desconectar], a[href*='modo=conectar']")?.scrollIntoView({ behavior: "smooth", block: "center" }));
  await espera(900);
  await mover(1320, 470);
  await espera(2400);

  // 12. Celular
  await rotulo("");
  await p.goto(`${BASE}/vitrina`);
  await p.waitForTimeout(800);
  const marco = p.frames().find((f) => f !== p.mainFrame());
  await marco.waitForSelector(".kpis", { timeout: 15000 });
  pos = { x: 1000, y: 700 };
  await espera(1500);
  const enMarco = (sel) => p.frameLocator("iframe").locator(sel);
  await marco.evaluate(() => window.scrollTo({ top: 520, behavior: "smooth" }));
  await espera(1600);
  await marco.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await espera(800);
  await clic(enMarco(".pestanas a[data-ruta=revisar]"), { espera: 1200 });
  await clic(enMarco(".item").nth(1), { espera: 2400 });
  await clic(enMarco(".pestanas a[data-ruta=clientes]"), { espera: 2200 });

  // 13. Cierre
  await p.evaluate(() => window.__portada(true, "Tu Cartera", "La cartera de tu estudio contable, siempre al día"));
  await espera(3600);

  await contexto.close();
  await navegador.close();
})();
