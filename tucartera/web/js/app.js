// Arranque, estructura de la aplicación, navegación y barra de instrucciones (⌘K).
import { api, configurar } from "./api.js";
import { LOGO, avatar, aviso, esc, icono } from "./ui.js";

const RUTAS = {
  hoy: () => import("./vistas/hoy.js"),
  revisar: () => import("./vistas/revisar.js"),
  clientes: () => import("./vistas/clientes.js"),
  cliente: () => import("./vistas/cliente.js"),
  vencimientos: () => import("./vistas/vencimientos.js"),
  documentos: () => import("./vistas/documentos.js"),
  cobranza: () => import("./vistas/cobranza.js"),
  tareas: () => import("./vistas/tareas.js"),
  automatizaciones: () => import("./vistas/automatizaciones.js"),
  actividad: () => import("./vistas/actividad.js"),
  ajustes: () => import("./vistas/ajustes.js"),
  plan: () => import("./vistas/plan.js"),
};

const NAV = [
  [null, [["hoy", "Hoy", "sol"], ["revisar", "Para revisar", "bandeja"]]],
  ["Cartera", [["clientes", "Clientes", "personas"], ["vencimientos", "Vencimientos", "calendario"],
    ["documentos", "Documentos", "documento"], ["cobranza", "Cobranza", "tarjeta"], ["tareas", "Tareas", "tareas"]]],
  ["Estudio", [["automatizaciones", "Automatizaciones", "ciclo"], ["actividad", "Actividad", "reloj"], ["plan", "Plan y uso", "medidor"], ["ajustes", "Ajustes", "engranaje"]]],
];

export const app = {
  sesion: null,
  navegar: (ruta) => { location.hash = ruta; },
  olvidarClientes: () => { clientesCache = null; },
  async refrescar() {
    try {
      app.sesion = await api.get("/api/sesion");
      const aviso = document.querySelector("[data-aviso-plan]");
      if (aviso) aviso.innerHTML = avisoPlan(app.sesion);
      document.querySelectorAll("[data-insignia]").forEach((el) => {
        el.textContent = app.sesion.para_revisar || "";
        el.classList.toggle("oculto", !app.sesion.para_revisar);
      });
    } catch { /* la vista mostrará el error */ }
  },
};

let limpiarVista = null;

try {
  const tema = localStorage.getItem("tucartera-tema");
  if (tema === "light" || tema === "dark") document.documentElement.setAttribute("data-theme", tema);
} catch { /* sin almacenamiento */ }

async function iniciar() {
  configurar({ sesionExpirada: () => { if (!app.sesion) return; app.sesion = null; mostrarIngreso(); } });
  try {
    app.sesion = await api.get("/api/sesion");
  } catch {
    return mostrarIngreso();
  }
  montar();
}

async function mostrarIngreso() {
  limpiarVista?.(); limpiarVista = null;
  const { render } = await import("./vistas/ingreso.js");
  document.body.innerHTML = "";
  const raiz = document.createElement("div");
  document.body.append(raiz);
  render(raiz, () => { location.hash = "#/hoy"; iniciar(); });
}

function avisoPlan(s) {
  const sus = s.suscripcion;
  if (!sus) return "";
  if (!sus.vigente) return `<a class="aviso-plan vencido" href="#/plan"><strong>Suscripción inactiva</strong>Las automatizaciones están en pausa. Activa tu plan.</a>`;
  if (sus.estado === "prueba") return `<a class="aviso-plan" href="#/plan"><strong>Prueba gratis · ${sus.dias_restantes === 1 ? "queda 1 día" : `quedan ${sus.dias_restantes} días`}</strong>Plan ${esc(sus.nombre)}. Elige tu plan cuando quieras.</a>`;
  return "";
}

function textoIA(s) {
  const ia = s?.ia;
  if (!ia || !ia.activa) return "Modo básico";
  if (!ia.permitido) return "Cuota de IA agotada · modo básico";
  return `IA: ${ia.porcentaje}% de tu cuota usada`;
}

function mostrarAvisoDeUrl() {
  const parametros = new URLSearchParams(location.search);
  const texto = parametros.get("aviso");
  if (texto) {
    history.replaceState(null, "", location.pathname + location.hash);
    setTimeout(() => aviso(texto, { duracion: 5000 }), 300);
  }
}

function montar() {
  const s = app.sesion;
  mostrarAvisoDeUrl();
  const enlaces = NAV.map(([grupo, items]) => `${grupo ? `<div class="nav-grupo">${grupo}</div>` : ""}` + items.map(([ruta, nombre, ico]) =>
    `<a href="#/${ruta}" data-ruta="${ruta}">${icono(ico)}<span>${nombre}</span>${ruta === "revisar"
      ? `<span class="insignia acento ${s.para_revisar ? "" : "oculto"}" data-insignia>${s.para_revisar || ""}</span>` : ""}</a>`).join("")).join("");
  document.body.innerHTML = `
  <div class="app">
    <aside class="barra" aria-label="Navegación">
      <div class="marca"><div class="marca-logo">${LOGO}</div><div class="truncar"><div class="marca-nombre">Tu Cartera</div>
        <div class="marca-estudio truncar">${esc(s.estudio.nombre)}</div></div></div>
      <button class="buscador" data-paleta>${icono("buscar", "s")}<span>Buscar o pedir algo</span><kbd>⌘K</kbd></button>
      <nav class="nav">${enlaces}</nav>
      <div data-aviso-plan>${avisoPlan(s)}</div>
      <a class="perfil" href="#/ajustes">${avatar(s.usuario, "m")}<div class="perfil-texto truncar">
        <div class="perfil-nombre truncar">${esc(s.usuario.nombre)}</div><div class="perfil-cargo truncar">${esc(s.usuario.cargo || "Equipo")}</div></div></a>
    </aside>
    <main class="contenido">
      <div class="barra-movil">
        <button class="boton fantasma icono" data-menu aria-label="Menú">${icono("menu")}</button>
        <div class="marca-nombre">Tu Cartera</div>
        <button class="boton fantasma icono" data-paleta aria-label="Buscar">${icono("buscar")}</button>
      </div>
      <div id="vista"></div>
    </main>
    <nav class="pestanas" aria-label="Accesos rápidos">
      <a href="#/hoy" data-ruta="hoy">${icono("sol")}Hoy</a>
      <a href="#/revisar" data-ruta="revisar">${icono("bandeja")}Revisar<span class="insignia acento ${s.para_revisar ? "" : "oculto"}" data-insignia>${s.para_revisar || ""}</span></a>
      <a href="#/clientes" data-ruta="clientes">${icono("personas")}Clientes</a>
      <a href="#/vencimientos" data-ruta="vencimientos">${icono("calendario")}Agenda</a>
      <button data-menu>${icono("menu")}Más</button>
    </nav>
  </div>`;
  const raiz = document.querySelector(".app");
  document.querySelectorAll("[data-menu]").forEach((b) => b.onclick = () => raiz.classList.toggle("menu-abierto"));
  raiz.addEventListener("click", (e) => {
    if (raiz.classList.contains("menu-abierto") && (e.target.closest(".nav a, .perfil") || e.target === raiz)) raiz.classList.remove("menu-abierto");
  });
  document.querySelectorAll("[data-paleta]").forEach((b) => b.onclick = abrirPaleta);
  enrutar();
}

async function enrutar() {
  if (!app.sesion || !document.getElementById("vista")) return;
  const partes = (location.hash.replace(/^#\/?/, "") || "hoy").split("/");
  let nombre = partes[0];
  let params = partes.slice(1);
  if (nombre === "clientes" && params[0]) { nombre = "cliente"; }
  if (!RUTAS[nombre]) { nombre = "hoy"; params = []; }
  const activa = nombre === "cliente" ? "clientes" : nombre;
  document.querySelectorAll("[data-ruta]").forEach((a) => {
    if (a.dataset.ruta === activa) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  limpiarVista?.();
  limpiarVista = null;
  const contenedor = document.getElementById("vista");
  const vista = document.createElement("div");
  vista.className = "vista";
  contenedor.replaceChildren(vista);
  const modulo = await RUTAS[nombre]();
  if (contenedor.firstChild !== vista) return; // se navegó a otra vista mientras cargaba
  try {
    limpiarVista = (await modulo.render(vista, params, app)) || null;
  } catch (e) {
    vista.innerHTML = `<div class="vacio"><div class="icono-cuadro rojo">${icono("alerta")}</div><h3>No pudimos cargar esta sección</h3><p>${esc(e.message)}</p></div>`;
  }
  const titulo = document.querySelector(".titulo, .detalle-titulo");
  document.title = `${titulo ? titulo.textContent.trim() + " · " : ""}Tu Cartera`;
}

window.addEventListener("hashchange", enrutar);

// ---------- Barra de instrucciones ----------
const DESTINOS = NAV.flatMap(([, items]) => items).map(([ruta, nombre, ico]) => ({ ruta, nombre, ico }));
const SUGERENCIAS = [
  "Pide los documentos que faltan",
  "Revisa los vencimientos de esta semana",
  "Prepara recordatorios de honorarios atrasados",
  "¿Qué clientes están en riesgo?",
];
let clientesCache = null;

function abrirPaleta() {
  if (document.querySelector(".paleta")) return;
  const velo = document.createElement("div");
  velo.className = "paleta-velo";
  const caja = document.createElement("div");
  caja.className = "paleta";
  caja.setAttribute("role", "dialog");
  caja.setAttribute("aria-label", "Buscar o pedir algo");
  caja.innerHTML = `<div class="paleta-entrada">${icono("buscar", "l")}<input type="text" placeholder="Busca un cliente o escribe un pedido" aria-label="Buscar o pedir algo" autocomplete="off" spellcheck="false"></div>
    <div class="paleta-resultados" role="listbox"></div>
    <div class="paleta-pie"><span>↑↓ para moverte</span><span>↵ para abrir o pedir</span><span>esc para cerrar</span><a href="#/plan" data-ir-plan style="margin-left:auto">${esc(textoIA(app.sesion))}</a></div>`;
  document.body.append(velo, caja);
  const entrada = caja.querySelector("input");
  const resultados = caja.querySelector(".paleta-resultados");
  let opciones = [];
  let indice = 0;
  const cerrar = () => { velo.remove(); caja.remove(); document.removeEventListener("keydown", teclas, true); };
  velo.onclick = cerrar;
  caja.querySelector("[data-ir-plan]").onclick = cerrar;

  if (!clientesCache) api.get("/api/clientes").then((d) => { clientesCache = d.clientes; pintar(); }).catch(() => {});

  function pintar() {
    const q = entrada.value.trim();
    const norma = (t) => t.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
    const nq = norma(q);
    opciones = [];
    let html = "";
    const grupo = (titulo, items) => {
      if (!items.length) return;
      html += `<div class="paleta-grupo">${titulo}</div>`;
      for (const it of items) {
        const i = opciones.push(it) - 1;
        html += `<button class="paleta-item" role="option" data-i="${i}">${it.html}</button>`;
      }
    };
    const clientes = (clientesCache || []).filter((c) => !q || norma(`${c.razon_social} ${c.nombre_corto} ${c.rut} ${c.giro}`).includes(nq)).slice(0, q ? 6 : 4);
    const pedir = () => grupo("Pedir", [{ accion: "pedir", texto: q, html: `<span class="icono-cuadro">${icono("enviar")}</span><span class="truncar">${esc(q)}</span><span class="t3">Enviar pedido</span>` }]);
    const parecePedido = q.split(/\s+/).length > 2;
    if (q && parecePedido) pedir();
    grupo("Clientes", clientes.map((c) => ({ accion: "ir", ruta: `#/clientes/${c.id}`, html: `<span class="icono-cuadro gris">${icono("edificio")}</span><span class="truncar">${esc(c.razon_social)}</span><span class="t3">${esc(c.rut)}</span>` })));
    if (q && !parecePedido) pedir();
    grupo("Ir a", DESTINOS.filter((d) => !q || norma(d.nombre).includes(nq)).slice(0, q ? 4 : 9).map((d) => ({ accion: "ir", ruta: `#/${d.ruta}`, html: `<span class="icono-cuadro gris">${icono(d.ico)}</span><span>${esc(d.nombre)}</span>` })));
    if (!q) grupo("Sugerencias", SUGERENCIAS.map((t) => ({ accion: "pedir", texto: t, html: `<span class="icono-cuadro">${icono("enviar")}</span><span>${esc(t)}</span>` })));
    resultados.innerHTML = html || `<div class="vacio pequeno">Sin resultados</div>`;
    indice = Math.min(indice, Math.max(0, opciones.length - 1));
    marcar();
  }
  function marcar() {
    resultados.querySelectorAll(".paleta-item").forEach((b) => {
      const sel = Number(b.dataset.i) === indice;
      b.setAttribute("aria-selected", sel);
      if (sel) b.scrollIntoView({ block: "nearest" });
    });
  }
  async function elegir(op) {
    if (!op) return;
    if (op.accion === "ir") { cerrar(); location.hash = op.ruta; return; }
    entrada.value = op.texto;
    entrada.disabled = true;
    resultados.remove();
    caja.querySelector(".paleta-pie").remove();
    const respuesta = document.createElement("div");
    respuesta.className = "paleta-respuesta";
    respuesta.innerHTML = `<div class="giro s" style="margin-top:4px"></div><div class="t2">Trabajando en tu pedido…</div>`;
    caja.append(respuesta);
    try {
      const r = await api.post("/api/instruccion", { texto: op.texto });
      await app.refrescar();
      const destino = r.navegar || (r.propuestas?.length ? "#/revisar" : null);
      respuesta.innerHTML = `<span class="icono-cuadro verde" style="width:26px;height:26px;border-radius:13px">${icono("check", "s")}</span>
        <div style="flex:1"><div>${esc(r.respuesta)}</div>${r.aviso ? `<div class="t2 pequeno" style="margin-top:6px">${esc(r.aviso)}</div>` : ""}<div class="acciones">
        ${destino ? `<button class="boton primario pequeno" data-ir="${esc(destino)}">${destino.startsWith("#/revisar") ? "Ver en Para revisar" : "Abrir"}</button>` : ""}
        <button class="boton pequeno" data-nuevo>Otro pedido</button></div></div>`;
      if (r.navegar && r.navegar.startsWith("#/clientes/")) { cerrar(); location.hash = r.navegar; return; }
    } catch (e) {
      respuesta.innerHTML = `<span class="icono-cuadro rojo" style="width:26px;height:26px;border-radius:13px">${icono("alerta", "s")}</span><div>${esc(e.message)}<div class="acciones"><button class="boton pequeno" data-nuevo>Intentar otro pedido</button></div></div>`;
    }
    respuesta.querySelector("[data-ir]")?.addEventListener("click", (e) => { cerrar(); location.hash = e.currentTarget.dataset.ir; });
    respuesta.querySelector("[data-nuevo]")?.addEventListener("click", () => { cerrar(); abrirPaleta(); });
    respuesta.querySelector("button")?.focus();
  }
  function teclas(e) {
    if (e.key === "Escape") { e.preventDefault(); cerrar(); return; }
    if (entrada.disabled) return;
    if (e.key === "ArrowDown") { e.preventDefault(); indice = Math.min(opciones.length - 1, indice + 1); marcar(); }
    if (e.key === "ArrowUp") { e.preventDefault(); indice = Math.max(0, indice - 1); marcar(); }
    if (e.key === "Enter") { e.preventDefault(); elegir(opciones[indice]); }
  }
  resultados.addEventListener("click", (e) => { const b = e.target.closest(".paleta-item"); if (b) elegir(opciones[Number(b.dataset.i)]); });
  resultados.addEventListener("mousemove", (e) => { const b = e.target.closest(".paleta-item"); if (b && Number(b.dataset.i) !== indice) { indice = Number(b.dataset.i); marcar(); } });
  entrada.addEventListener("input", () => { indice = 0; pintar(); });
  document.addEventListener("keydown", teclas, true);
  pintar();
  entrada.focus();
}

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k" && app.sesion) { e.preventDefault(); abrirPaleta(); }
  else if (e.key === "/" && app.sesion && !e.target.closest("input, textarea, select, [contenteditable]")) { e.preventDefault(); abrirPaleta(); }
});

app.abrirPaleta = abrirPaleta;

// Instalación como app (PC y celular).
app.instalacion = null;
window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); app.instalacion = e; });
if ("serviceWorker" in navigator && window.isSecureContext) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch(() => {}));
}
iniciar();
