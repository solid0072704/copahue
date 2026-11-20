// Para revisar: lo que prepararon las automatizaciones, con vista dividida al estilo de Mail.
import { api } from "../api.js";
import { COLOR_AGENTE, avatar, aviso, cargando, esc, fechaLarga, icono, iconoAgente, relativa, vacio } from "../ui.js";

const FILTROS = [["todo", "Todo"], ["correo", "Correos"], ["tarea", "Tareas"], ["historial", "Historial"]];

export async function render(el, params, app) {
  el.className = "vista ancha";
  el.innerHTML = cargando();
  let filtro = "todo";
  let pendientes = (await api.get("/api/propuestas")).propuestas;
  let historial = null;
  let seleccion = params[0] ? Number(params[0]) : null;

  el.innerHTML = `<div class="dividida">
    <section class="panel-lista" aria-label="Propuestas">
      <div class="panel-lista-cab">
        <h1 class="titulo">Para revisar</h1>
        <p class="t2 pequeno" data-cuenta style="margin-top:4px"></p>
        <div class="segmentado" role="group" aria-label="Filtrar">${FILTROS.map(([v, t]) => `<button data-filtro="${v}" aria-pressed="${v === filtro}">${t}</button>`).join("")}</div>
      </div>
      <div class="bandeja" role="listbox" aria-label="Propuestas"></div>
    </section>
    <section class="detalle" aria-live="polite"></section>
  </div>`;
  const dividida = el.querySelector(".dividida");
  const bandeja = el.querySelector(".bandeja");
  const detalle = el.querySelector(".detalle");
  const cuenta = el.querySelector("[data-cuenta]");

  const visibles = () => filtro === "historial" ? (historial || []) : pendientes.filter((p) => filtro === "todo" || p.tipo === filtro);

  function pintarLista() {
    const lista = visibles();
    const altas = pendientes.filter((p) => p.prioridad === 1).length;
    cuenta.textContent = pendientes.length
      ? `${pendientes.length} pendiente${pendientes.length === 1 ? "" : "s"}${altas ? ` · ${altas} con prioridad alta` : ""}`
      : "Todo revisado";
    if (filtro === "historial" && !historial) { bandeja.innerHTML = cargando(); return; }
    if (!lista.length) {
      bandeja.innerHTML = filtro === "historial" ? `<div class="vacio pequeno">Aún no hay decisiones registradas.</div>` : "";
      return;
    }
    bandeja.innerHTML = lista.map((p) => `
      <button class="item" role="option" data-id="${p.id}" aria-selected="${p.id === seleccion}">
        ${p.prioridad === 1 && p.estado === "pendiente" ? `<span class="prioridad" aria-label="Prioridad alta"></span>` : ""}
        <span class="icono-cuadro ${COLOR_AGENTE[p.agente] || ""}">${iconoAgente(p.agente_icono)}</span>
        <span class="item-cuerpo">
          <span class="item-fuente"><span class="truncar">${esc(p.agente_nombre)}</span><span>${esc(relativa(p.estado === "pendiente" ? p.creada : p.resuelta))}</span></span>
          <div class="item-titulo truncar">${esc(p.titulo)}</div>
          <div class="item-sub">${p.estado === "pendiente" ? esc(p.resumen) : `${esc(p.cliente?.nombre || "")} · ${esc(p.resultado)}`}</div>
        </span>
      </button>`).join("");
  }

  function pintarDetalle() {
    const p = visibles().find((x) => x.id === seleccion);
    dividida.classList.toggle("con-detalle", !!p);
    if (!p) {
      detalle.innerHTML = pendientes.length || filtro === "historial"
        ? `<div class="vacio" style="margin:auto"><div class="icono-cuadro gris">${icono("bandeja")}</div><h3>Elige una propuesta</h3><p>Revisa el borrador, ajústalo si quieres y apruébalo.</p></div>`
        : `<div style="margin:auto">${vacio("check", "Todo al día", "No hay nada pendiente de revisión. Las automatizaciones dejarán aquí lo próximo.")}</div>`;
      return;
    }
    const d = p.datos;
    const resuelta = p.estado !== "pendiente";
    const cuerpo = p.tipo === "correo" ? `
      <div class="correo">
        <div class="correo-campo"><label for="c-para">Para</label><input id="c-para" name="para" value="${esc(d.para)}" ${resuelta ? "readonly" : ""}></div>
        <div class="correo-campo"><label for="c-asunto">Asunto</label><input id="c-asunto" class="asunto" name="asunto" value="${esc(d.asunto)}" ${resuelta ? "readonly" : ""}></div>
        <textarea name="cuerpo" aria-label="Cuerpo del correo" ${resuelta ? "readonly" : ""}>${esc(d.cuerpo)}</textarea>
        <div class="correo-pie">${icono("info", "xs")}${d.responder_a ? `Las respuestas llegarán a ${esc(d.responder_a.replace(/\+[^@]+@/, "@"))}` : "Las respuestas llegarán al correo del estudio"}${d.redaccion_asistida ? " · Redacción pulida" : ""}</div>
      </div>` : `
      <div class="tarea-propuesta">
        <span class="icono-cuadro grande ${COLOR_AGENTE[p.agente] || ""}">${icono("tareas", "l")}</span>
        <div style="min-width:0">
          <h3>${esc(d.tarea?.titulo || p.titulo)}</h3>
          ${d.tarea?.detalle ? `<p class="t2" style="margin-top:4px">${esc(d.tarea.detalle)}</p>` : ""}
          <div class="tarea-meta">
            ${d.tarea?.responsable ? `<span>${avatar(d.tarea.responsable)}${esc(d.tarea.responsable.nombre)}</span>` : ""}
            ${d.tarea?.vence ? `<span>${icono("calendario", "s")}${esc(fechaLarga(d.tarea.vence))}</span>` : ""}
          </div>
        </div>
      </div>`;
    detalle.innerHTML = `
      <div class="detalle-cuerpo">
        <button class="boton fantasma volver-movil" data-volver style="margin:-6px 0 10px -8px;display:none">${icono("izq", "s")}Para revisar</button>
        <div class="detalle-fuente"><span class="icono-cuadro ${COLOR_AGENTE[p.agente] || ""}" style="width:22px;height:22px;border-radius:6px">${iconoAgente(p.agente_icono)}</span>
          ${esc(p.agente_nombre)} · ${esc(relativa(p.creada))}${p.prioridad === 1 ? ` · <span style="color:var(--rojo)">Prioridad alta</span>` : ""}</div>
        <h2 class="detalle-titulo">${esc(p.titulo)}</h2>
        ${p.cliente ? `<a class="detalle-cliente" href="#/clientes/${p.cliente.id}">${esc(p.cliente.razon_social)}</a>` : `<div class="detalle-cliente">${esc(p.resumen)}</div>`}
        ${p.motivos?.length ? `<div class="motivos"><div class="motivos-titulo">Por qué</div><ul>${p.motivos.map((m) => `<li>${icono("check", "xs")}<span>${esc(m)}</span></li>`).join("")}</ul></div>` : `<div style="height:22px"></div>`}
        ${cuerpo}
        ${resuelta ? `<div class="motivos" style="margin-top:18px"><div class="motivos-titulo">${p.estado === "aprobada" ? "Aprobada" : p.estado === "retirada" ? "Retirada automáticamente" : "Descartada"}${p.resuelta_por ? ` por ${esc(p.resuelta_por.nombre)}` : ""} · ${esc(relativa(p.resuelta))}</div><div>${esc(p.resultado)}</div></div>` : ""}
      </div>
      ${resuelta ? "" : `<div class="barra-acciones">
        <div class="atajos"><span><kbd>↑</kbd> <kbd>↓</kbd> moverse</span><span><kbd>⌘</kbd> <kbd>↵</kbd> aprobar</span></div>
        <div class="acciones" style="margin-left:auto">
          <button class="boton" data-descartar>Descartar</button>
          <button class="boton primario" data-aprobar>${p.tipo === "correo" ? `${icono("enviar", "s")}Aprobar y enviar` : `${icono("check", "s")}Aprobar y asignar`}</button>
        </div></div>`}`;
    if (window.matchMedia("(max-width: 860px)").matches) detalle.querySelector("[data-volver]").style.display = "inline-flex";
    detalle.querySelector("[data-volver]").onclick = () => { seleccion = null; history.replaceState(null, "", "#/revisar"); pintarLista(); pintarDetalle(); };
    detalle.querySelector("[data-aprobar]")?.addEventListener("click", () => aprobar(p));
    detalle.querySelector("[data-descartar]")?.addEventListener("click", () => descartar(p));
    detalle.scrollTop = 0;
  }

  function siguienteDe(p) {
    const lista = visibles();
    const i = lista.findIndex((x) => x.id === p.id);
    return (lista[i + 1] || lista[i - 1] || null)?.id ?? null;
  }

  function quitar(p) {
    const siguiente = siguienteDe(p);
    pendientes = pendientes.filter((x) => x.id !== p.id);
    historial = null;
    seleccion = window.matchMedia("(max-width: 860px)").matches ? null : siguiente;
    history.replaceState(null, "", seleccion ? `#/revisar/${seleccion}` : "#/revisar");
    pintarLista();
    pintarDetalle();
    app.refrescar();
  }

  async function aprobar(p) {
    const boton = detalle.querySelector("[data-aprobar]");
    boton.disabled = true;
    const cambios = {};
    if (p.tipo === "correo") {
      for (const campo of ["para", "asunto", "cuerpo"]) cambios[campo] = detalle.querySelector(`[name=${campo}]`).value;
    }
    try {
      const r = await api.post(`/api/propuestas/${p.id}/aprobar`, cambios);
      aviso(r.resultado);
      quitar(p);
    } catch (e) {
      aviso(e.message, { error: true });
      boton.disabled = false;
    }
  }

  async function descartar(p) {
    try {
      await api.post(`/api/propuestas/${p.id}/descartar`);
      aviso("Propuesta descartada");
      quitar(p);
    } catch (e) { aviso(e.message, { error: true }); }
  }

  el.querySelector(".segmentado").addEventListener("click", async (e) => {
    const b = e.target.closest("[data-filtro]");
    if (!b) return;
    filtro = b.dataset.filtro;
    el.querySelectorAll("[data-filtro]").forEach((x) => x.setAttribute("aria-pressed", x === b));
    seleccion = null;
    pintarLista();
    pintarDetalle();
    if (filtro === "historial" && !historial) {
      historial = (await api.get("/api/propuestas?estado=resueltas")).propuestas;
      if (filtro === "historial") pintarLista();
    }
  });
  bandeja.addEventListener("click", (e) => {
    const b = e.target.closest(".item");
    if (!b) return;
    seleccion = Number(b.dataset.id);
    history.replaceState(null, "", `#/revisar/${seleccion}`);
    bandeja.querySelectorAll(".item").forEach((x) => x.setAttribute("aria-selected", x === b));
    pintarDetalle();
  });

  const teclas = (e) => {
    if (document.querySelector(".paleta, .velo")) return;
    const enCampo = e.target.closest("input, textarea, select");
    const lista = visibles();
    const i = lista.findIndex((x) => x.id === seleccion);
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      const p = lista[i];
      if (p && p.estado === "pendiente") { e.preventDefault(); aprobar(p); }
      return;
    }
    if (enCampo) return;
    if (e.key === "ArrowDown" || e.key === "j" || e.key === "ArrowUp" || e.key === "k") {
      e.preventDefault();
      const n = e.key === "ArrowDown" || e.key === "j" ? Math.min(lista.length - 1, i + 1) : Math.max(0, i - 1);
      if (lista[n]) {
        seleccion = lista[n].id;
        history.replaceState(null, "", `#/revisar/${seleccion}`);
        pintarLista();
        pintarDetalle();
        bandeja.querySelector(`[data-id="${seleccion}"]`)?.scrollIntoView({ block: "nearest" });
      }
    }
  };
  document.addEventListener("keydown", teclas);

  if (!seleccion && pendientes.length && !window.matchMedia("(max-width: 860px)").matches) seleccion = pendientes[0].id;
  pintarLista();
  pintarDetalle();
  return () => document.removeEventListener("keydown", teclas);
}
