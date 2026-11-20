// Utilidades de interfaz: escape, formato, iconos, avatares, avisos, hojas y gráficos.

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);

// ---------- Iconos (trazos de 24 px, al estilo de los símbolos del sistema) ----------
const P = {
  sol: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4"/>',
  bandeja: '<path d="M4.5 13 6.6 6.3A2 2 0 0 1 8.5 5h7a2 2 0 0 1 1.9 1.3l2.1 6.7v4.5A1.5 1.5 0 0 1 18 19H6a1.5 1.5 0 0 1-1.5-1.5Z"/><path d="M4.5 13H8l1.5 2.5h5L16 13h3.5"/>',
  personas: '<circle cx="9.5" cy="8" r="3.3"/><path d="M3.5 19.5v-.8A4.7 4.7 0 0 1 8.2 14h2.6a4.7 4.7 0 0 1 4.7 4.7v.8"/><path d="M15.5 4.9a3.3 3.3 0 0 1 0 6.2M17.8 14.2a4.7 4.7 0 0 1 2.7 4.3v1"/>',
  calendario: '<rect x="3.5" y="5" width="17" height="15.5" rx="2.6"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
  documento: '<path d="M14 3.5H7.5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8Z"/><path d="M14 3.5V8h4.5M9 13h6M9 16.5h4"/>',
  tarjeta: '<rect x="3" y="5.5" width="18" height="13" rx="2.6"/><path d="M3 10h18M7 15h3"/>',
  tareas: '<path d="M10 6.5h10M10 12h10M10 17.5h10"/><path d="m3.8 6.5 1.3 1.3 2.4-2.6M3.8 12l1.3 1.3 2.4-2.6M3.8 17.5l1.3 1.3 2.4-2.6"/>',
  ciclo: '<path d="M19.5 11A7.5 7.5 0 0 0 6.2 6.6L4.5 8.5"/><path d="M4.5 4.5v4h4"/><path d="M4.5 13a7.5 7.5 0 0 0 13.3 4.4l1.7-1.9"/><path d="M19.5 19.5v-4h-4"/>',
  reloj: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
  engranaje: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/>',
  grafico: '<path d="M4.5 19.5v-6M9.5 19.5v-11M14.5 19.5v-8M19.5 19.5V5"/>',
  pulso: '<path d="M3 12.5h4l2.5-6 4 12 2.5-6H21"/>',
  informe: '<rect x="5" y="3.5" width="14" height="17" rx="2.2"/><path d="M8.5 8.5h7M8.5 12h7M8.5 15.5h4"/>',
  comando: '<path d="M9 6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3Z"/>',
  buscar: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
  mas: '<path d="M12 5v14M5 12h14"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  x: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
  der: '<path d="m9.5 5.5 6.5 6.5-6.5 6.5"/>',
  izq: '<path d="m14.5 5.5-6.5 6.5 6.5 6.5"/>',
  abajo: '<path d="m5.5 9.5 6.5 6.5 6.5-6.5"/>',
  sobre: '<rect x="3.5" y="5.5" width="17" height="13" rx="2.6"/><path d="m4.5 7.5 7.5 5.5 7.5-5.5"/>',
  telefono: '<path d="M20 15.9v2.6a1.8 1.8 0 0 1-2 1.8A16.6 16.6 0 0 1 3.7 6a1.8 1.8 0 0 1 1.8-2h2.6a1.8 1.8 0 0 1 1.8 1.5c.1.9.4 1.7.7 2.5a1.8 1.8 0 0 1-.4 1.9l-1.1 1.1a14 14 0 0 0 5.9 5.9l1.1-1.1a1.8 1.8 0 0 1 1.9-.4c.8.3 1.6.6 2.5.7a1.8 1.8 0 0 1 1.5 1.8Z"/>',
  alerta: '<path d="M10.3 4.6 2.9 17.5A2 2 0 0 0 4.6 20.5h14.8a2 2 0 0 0 1.7-3L13.7 4.6a2 2 0 0 0-3.4 0Z"/><path d="M12 9.5v4M12 17h.01"/>',
  enviar: '<path d="M20.5 3.5 10 14M20.5 3.5 14 20.5l-4-6.5-6.5-4Z"/>',
  salir: '<path d="M14.5 4.5h3a1.5 1.5 0 0 1 1.5 1.5v12a1.5 1.5 0 0 1-1.5 1.5h-3M9.5 16.5 5 12l4.5-4.5M5 12h10.5"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  persona: '<circle cx="12" cy="8" r="3.6"/><path d="M5 20a7 7 0 0 1 14 0"/>',
  edificio: '<path d="M4.5 20.5V5A1.5 1.5 0 0 1 6 3.5h8A1.5 1.5 0 0 1 15.5 5v15.5M15.5 9.5H18a1.5 1.5 0 0 1 1.5 1.5v9.5M3 20.5h18M8 7.5h4M8 11h4M8 14.5h4"/>',
  flecha: '<path d="M7 17 17 7M8.5 7H17v8.5"/>',
  luna: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z"/>',
  lapiz: '<path d="M15.5 5.5l3 3M4.5 19.5l1-4L16 5a2.1 2.1 0 0 1 3 3L8.5 18.5Z"/>',
  basurero: '<path d="M4.5 7h15M9.5 7V5a1.5 1.5 0 0 1 1.5-1.5h2A1.5 1.5 0 0 1 14.5 5v2M6.5 7l.8 11.6A2 2 0 0 0 9.3 20.5h5.4a2 2 0 0 0 2-1.9L17.5 7"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8h.01"/>',
  medidor: '<path d="M4.5 17.5a8.5 8.5 0 1 1 15 0"/><path d="m12 13 3.5-4"/><circle cx="12" cy="13.5" r="1.2"/>',
  descargar: '<path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M5 19.5h14"/>',
  escudo: '<path d="M12 3.5 5 6v5.5c0 4.3 2.9 7.6 7 9 4.1-1.4 7-4.7 7-9V6Z"/><path d="m9 12 2 2 4-4"/>',
};
const ICONO_AGENTE = { calendario: "calendario", documento: "documento", grafico: "grafico", tarjeta: "tarjeta", pulso: "pulso", informe: "informe", comando: "comando" };
export const COLOR_AGENTE = { vencimientos: "rojo", documentos: "azul", control: "indigo", cobranza: "verde", cartera: "naranja", informes: "turquesa", asistente: "gris" };

export function icono(nombre, clase = "") {
  return `<svg class="ico ${clase}" viewBox="0 0 24 24" aria-hidden="true">${P[nombre] || P.info}</svg>`;
}
export const iconoAgente = (n) => icono(ICONO_AGENTE[n] || "ciclo");

// Logotipo de Tu Cartera: cuatro casillas, la cartera de clientes en orden.
export const LOGO = `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="8" height="8" rx="2.2" fill="currentColor"/><rect x="13" y="3" width="8" height="8" rx="2.2" fill="currentColor" opacity=".42"/><rect x="3" y="13" width="8" height="8" rx="2.2" fill="currentColor" opacity=".42"/><rect x="13" y="13" width="8" height="8" rx="2.2" fill="currentColor"/></svg>`;

// ---------- Formato ----------
const MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sept", "oct", "nov", "dic"];
const MESES_LARGOS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"];
const DIAS = ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"];
const DIAS_LARGOS = ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"];
export { MESES_LARGOS, DIAS_LARGOS };

export const clp = (n) => (n < 0 ? "−" : "") + "$" + Math.abs(Math.round(n || 0)).toLocaleString("es-CL");
export function clpCorto(n) {
  const v = Math.abs(n || 0);
  if (v >= 1e9) return "$" + (n / 1e9).toLocaleString("es-CL", { maximumFractionDigits: 1 }) + " mil M";
  if (v >= 1e6) return "$" + (n / 1e6).toLocaleString("es-CL", { maximumFractionDigits: v < 1e7 ? 2 : 1 }) + " M";
  if (v >= 1e3) return "$" + Math.round(n / 1e3).toLocaleString("es-CL") + " mil";
  return clp(n);
}
export const fecha = (iso) => {
  if (!iso) return "";
  const [a, m, d] = iso.slice(0, 10).split("-").map(Number);
  return new Date(a, m - 1, d);
};
export const isoDia = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
export const fechaCorta = (iso) => { const d = fecha(iso); return d ? `${d.getDate()} ${MESES[d.getMonth()]}` : ""; };
export const fechaDia = (iso) => { const d = fecha(iso); return d ? `${DIAS[d.getDay()]} ${d.getDate()} ${MESES[d.getMonth()]}` : ""; };
export const fechaLarga = (iso) => { const d = fecha(iso); return d ? `${DIAS_LARGOS[d.getDay()]} ${d.getDate()} de ${MESES_LARGOS[d.getMonth()]}` : ""; };
export const diasEntre = (desde, hasta) => Math.round((fecha(hasta) - fecha(desde)) / 86400000);

export function cuando(iso, hoy) {
  const n = diasEntre(hoy, iso);
  if (n === 0) return "Hoy";
  if (n === 1) return "Mañana";
  if (n === -1) return "Ayer";
  if (n < 0) return `Hace ${-n} días`;
  if (n < 7) return DIAS_LARGOS[fecha(iso).getDay()].replace(/^./, (c) => c.toUpperCase());
  return fechaCorta(iso);
}

export function relativa(isoHora) {
  if (!isoHora) return "";
  const t = new Date(isoHora);
  const s = (Date.now() - t.getTime()) / 1000;
  if (s < 60) return "Ahora";
  if (s < 3600) return `Hace ${Math.floor(s / 60)} min`;
  if (s < 86400) return `Hace ${Math.floor(s / 3600)} h`;
  if (s < 172800) return "Ayer";
  if (s < 604800) return `Hace ${Math.floor(s / 86400)} días`;
  return `${t.getDate()} ${MESES[t.getMonth()]}`;
}

export const plural = (n, uno, varios) => `${n} ${n === 1 ? uno : varios}`;
export const miles = (n) => Math.round(n || 0).toLocaleString("es-CL");
export function tokensCorto(n) {
  if (n >= 1e6) return (n / 1e6).toLocaleString("es-CL", { maximumFractionDigits: 1 }) + " M";
  if (n >= 1e3) return Math.round(n / 1e3).toLocaleString("es-CL") + " mil";
  return miles(n);
}

// ---------- Personas y clientes ----------
export const iniciales = (nombre) => (nombre || "?").split(/\s+/).filter((p) => p && p[0] === p[0].toUpperCase()).slice(0, 2).map((p) => p[0]).join("") || "?";

export function avatar(persona, tam = "") {
  if (!persona) return "";
  return `<span class="avatar ${tam}" style="background:${esc(persona.color || "#8e8e93")}" title="${esc(persona.nombre)}">${esc(iniciales(persona.nombre))}</span>`;
}

const TONOS = [
  ["#e8f0fe", "#1a56b8"], ["#e7f6ec", "#1d7a3a"], ["#fff1e0", "#a24d00"], ["#fde8ec", "#b4233c"],
  ["#e6f6f8", "#08707f"], ["#efeefe", "#4643b3"], ["#f3efe7", "#7a5b2e"], ["#eef0f3", "#4b5563"],
];
export function monograma(nombre, tam = "") {
  let h = 0;
  for (const c of nombre || "") h = (h * 31 + c.charCodeAt(0)) >>> 0;
  const [bg, fg] = TONOS[h % TONOS.length];
  const letras = (nombre || "?").replace(/^(el|la|los|las)\s+/i, "").split(/\s+/).filter(Boolean).slice(0, 2).map((p) => p[0]).join("").toUpperCase();
  return `<span class="monograma ${tam}" style="background:${bg};color:${fg}">${esc(letras)}</span>`;
}

export function anilloSalud(salud, grande = false) {
  if (!salud || salud.puntaje == null) return `<span class="pill">Pausado</span>`;
  const r = 9, c = 2 * Math.PI * r, v = Math.max(0.02, salud.puntaje / 100);
  return `<span class="salud ${esc(salud.nivel)} ${grande ? "grande" : ""}" title="${esc((salud.motivos || []).join(" · ") || "Al día")}">
    <svg viewBox="0 0 24 24"><circle class="pista" cx="12" cy="12" r="${r}" fill="none" stroke-width="3"/>
    <circle cx="12" cy="12" r="${r}" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"
      stroke-dasharray="${(c * v).toFixed(2)} ${c.toFixed(2)}"/></svg>${grande ? "" : esc(salud.puntaje)}</span>`;
}
export const NIVEL = { ok: "Al día", atencion: "Atención", riesgo: "En riesgo", pausado: "Pausado" };

const ESTADOS_OBLIGACION = {
  pendiente: ["", "Sin comenzar"], preparacion: ["azul", "En preparación"],
  revision: ["naranja", "En revisión"], presentada: ["verde", "Presentada"],
};
export function estadoObligacion(o) {
  if (o.atrasada) return `<span class="pill rojo">Atrasada</span>`;
  const [clase, texto] = ESTADOS_OBLIGACION[o.estado] || ["", o.estado];
  return `<span class="pill ${clase}">${texto}</span>`;
}
export const ESTADOS_OBL = ESTADOS_OBLIGACION;
export const etiquetaTipo = (o) => `<span class="etiqueta-tipo ${esc(o.tipo)}">${esc(o.nombre || o.tipo)}</span>`;

// ---------- Avisos ----------
let contenedorAvisos;
export function aviso(texto, { error = false, enlace = null, duracion = 3600 } = {}) {
  if (!contenedorAvisos) {
    contenedorAvisos = document.createElement("div");
    contenedorAvisos.className = "avisos";
    contenedorAvisos.setAttribute("role", "status");
    contenedorAvisos.setAttribute("aria-live", "polite");
    document.body.append(contenedorAvisos);
  }
  const el = document.createElement("div");
  el.className = `aviso ${error ? "error" : ""}`;
  el.innerHTML = `${icono(error ? "alerta" : "check", "s")}<span>${esc(texto)}${enlace ? ` <a href="${esc(enlace.href)}">${esc(enlace.texto)}</a>` : ""}</span>`;
  contenedorAvisos.append(el);
  setTimeout(() => { el.classList.add("saliendo"); setTimeout(() => el.remove(), 260); }, duracion);
}

// ---------- Hoja modal ----------
export function hoja({ titulo, contenido, aceptar = "Listo", cancelar = "Cancelar", alAceptar, alAbrir }) {
  const velo = document.createElement("div");
  velo.className = "velo";
  velo.innerHTML = `<div class="hoja" role="dialog" aria-modal="true" aria-label="${esc(titulo)}">
    <div class="hoja-cab"><button class="boton fantasma" data-cerrar>${esc(cancelar)}</button><h2>${esc(titulo)}</h2>
    <button class="boton primario pequeno" data-aceptar>${esc(aceptar)}</button></div>
    <div class="hoja-cuerpo">${contenido}</div></div>`;
  const previo = document.activeElement;
  const cerrar = () => { velo.remove(); document.removeEventListener("keydown", tecla); previo?.focus?.(); };
  const tecla = (e) => { if (e.key === "Escape") cerrar(); };
  velo.addEventListener("mousedown", (e) => { if (e.target === velo) cerrar(); });
  velo.querySelector("[data-cerrar]").onclick = cerrar;
  const boton = velo.querySelector("[data-aceptar]");
  boton.onclick = async () => {
    boton.disabled = true;
    try {
      const ok = alAceptar ? await alAceptar(velo.querySelector(".hoja-cuerpo")) : true;
      if (ok !== false) cerrar();
    } finally { boton.disabled = false; }
  };
  document.addEventListener("keydown", tecla);
  document.body.append(velo);
  alAbrir?.(velo.querySelector(".hoja-cuerpo"));
  velo.querySelector("input, textarea, select")?.focus();
  return cerrar;
}

// ---------- Gráficos ----------
export function barras(series, etiquetas, { alto = 170, ancho = 640, formato = clpCorto } = {}) {
  const izq = 44, abajo = 22, arriba = 8;
  const max = Math.max(1, ...series.flatMap((s) => s.valores));
  const escala = niceMax(max);
  const n = etiquetas.length;
  const grupo = (ancho - izq) / n;
  const barra = Math.min(16, (grupo - 10) / series.length);
  const h = alto - abajo - arriba;
  let svg = `<svg class="grafico" viewBox="0 0 ${ancho} ${alto}" role="img" aria-label="Gráfico de barras">`;
  for (let i = 0; i <= 3; i++) {
    const y = arriba + h - (h * i) / 3;
    svg += `<line class="rejilla-linea" x1="${izq}" x2="${ancho}" y1="${y}" y2="${y}"/>`;
    svg += `<text x="${izq - 8}" y="${y + 3.5}" text-anchor="end">${esc(formato((escala * i) / 3))}</text>`;
  }
  etiquetas.forEach((et, i) => {
    const x0 = izq + grupo * i + (grupo - barra * series.length - 3 * (series.length - 1)) / 2;
    series.forEach((s, k) => {
      const v = s.valores[i] || 0;
      const bh = Math.max(v ? 2 : 0, (h * v) / escala);
      const x = x0 + k * (barra + 3);
      svg += `<rect class="${s.clase}" x="${x.toFixed(1)}" y="${(arriba + h - bh).toFixed(1)}" width="${barra.toFixed(1)}" height="${bh.toFixed(1)}" rx="3"><title>${esc(s.nombre)} ${esc(et)}: ${esc(clp(v))}</title></rect>`;
    });
    svg += `<text x="${(izq + grupo * i + grupo / 2).toFixed(1)}" y="${alto - 6}" text-anchor="middle">${esc(et)}</text>`;
  });
  return svg + "</svg>";
}
function niceMax(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  return (m <= 1.5 ? 1.5 : m <= 3 ? 3 : m <= 6 ? 6 : 10) * p;
}

export const cargando = () => `<div class="cargando"><div class="giro"></div></div>`;
export function vacio(iconoNombre, titulo, texto, clase = "verde") {
  return `<div class="vacio"><div class="icono-cuadro ${clase}">${icono(iconoNombre)}</div><h3>${esc(titulo)}</h3><p>${esc(texto)}</p></div>`;
}
