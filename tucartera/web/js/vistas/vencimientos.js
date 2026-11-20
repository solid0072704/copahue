// Calendario tributario del estudio: vista de mes al estilo de Calendario y lista del día.
import { api } from "../api.js";
import { ESTADOS_OBL, avatar, aviso, cargando, clp, esc, estadoObligacion, etiquetaTipo, fecha, fechaLarga, icono, isoDia, plural } from "../ui.js";

const SEMANA = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"];

export async function render(el, params, app) {
  el.innerHTML = cargando();
  let mes = /^\d{4}-\d{2}$/.test(params[0] || "") ? params[0] : "";
  let tipo = "todos";
  let d = await api.get(`/api/vencimientos${mes ? `?mes=${mes}` : ""}`);
  const hoy = app.sesion.hoy;
  let dia = d.mes === hoy.slice(0, 7) ? hoy : null;

  async function irA(nuevo) {
    mes = nuevo;
    history.replaceState(null, "", `#/vencimientos/${mes}`);
    d = await api.get(`/api/vencimientos?mes=${mes}`);
    dia = d.mes === hoy.slice(0, 7) ? hoy : null;
    pintar();
  }

  function pintar() {
    const lista = d.obligaciones.filter((o) => tipo === "todos" || o.tipo === tipo);
    const delMes = lista.filter((o) => o.vence.slice(0, 7) === d.mes);
    const pendientes = delMes.filter((o) => o.estado !== "presentada").length;
    const atrasadas = lista.filter((o) => o.atrasada);
    const porDia = {};
    for (const o of delMes) (porDia[o.vence] ||= []).push(o);
    const feriados = new Set(d.feriados);

    const primero = fecha(`${d.mes}-01`);
    const inicio = new Date(primero);
    inicio.setDate(1 - ((primero.getDay() + 6) % 7));
    const diasMes = new Date(primero.getFullYear(), primero.getMonth() + 1, 0).getDate();
    const total = Math.ceil(((primero.getDay() + 6) % 7 + diasMes) / 7) * 7;
    let celdas = "";
    for (let i = 0; i < total; i++) {
      const f = new Date(inicio);
      f.setDate(inicio.getDate() + i);
      const iso = isoDia(f);
      const items = porDia[iso] || [];
      const grupos = {};
      for (const o of items) (grupos[o.tipo] ||= []).push(o);
      const fuera = iso.slice(0, 7) !== d.mes;
      const inhabil = f.getDay() === 0 || f.getDay() === 6 || feriados.has(iso);
      celdas += `<button class="cal-celda ${fuera ? "fuera" : ""} ${inhabil ? "inhabil" : ""} ${iso === hoy ? "hoy" : ""}" data-dia="${iso}" aria-pressed="${iso === dia}" aria-label="${esc(fechaLarga(iso))}: ${items.length} vencimientos">
        <span class="cal-num">${f.getDate()}</span>
        ${feriados.has(iso) ? `<span class="cal-feriado">Feriado</span>` : ""}
        ${Object.entries(grupos).map(([t, os]) => {
          const listas = os.filter((o) => o.estado === "presentada").length;
          return `<span class="cal-evento ${t} ${listas === os.length ? "listo" : ""}"><span>${esc(d.tipos[t].nombre)}</span><span>${listas}/${os.length}</span></span>`;
        }).join("")}
      </button>`;
    }

    const seleccion = dia ? lista.filter((o) => o.vence === dia) : [];
    const bloque = (titulo, items, sub = "") => `<section class="tarjeta" style="margin-top:18px">
      <div class="tarjeta-cab"><h2 class="tarjeta-titulo">${esc(titulo)}</h2><span class="t2 pequeno">${esc(sub)}</span></div>
      ${items.length ? `<ul class="lista">${items.map((o) => `<li class="fila">
        ${etiquetaTipo(o)}
        <a class="fila-cuerpo" href="#/clientes/${o.cliente_id}/declaraciones" style="color:inherit;text-decoration:none">
          <div class="fila-titulo truncar">${esc(o.nombre_corto || o.razon_social)}</div>
          <div class="fila-sub truncar">${esc(o.descripcion)} · ${esc(o.periodo_nombre)}${o.monto ? ` · ${esc(clp(o.monto))}` : ""}</div></a>
        ${o.responsable ? `<span class="ocultar-movil">${avatar(o.responsable)}</span>` : ""}
        ${o.estado === "presentada" ? estadoObligacion(o) : `${o.atrasada ? estadoObligacion(o) : ""}<select class="selector" data-obligacion="${o.id}" aria-label="Estado de ${esc(o.nombre)} de ${esc(o.nombre_corto || o.razon_social)}">
          ${Object.entries(ESTADOS_OBL).map(([v, [, t]]) => `<option value="${v}" ${v === o.estado ? "selected" : ""}>${t}</option>`).join("")}</select>`}
      </li>`).join("")}</ul>` : `<div class="vacio pequeno">Sin vencimientos este día.</div>`}</section>`;

    el.innerHTML = `
      <header class="encabezado">
        <div class="encabezado-texto"><h1 class="titulo">Vencimientos</h1>
          <p class="subtitulo">${esc(plural(pendientes, "declaración pendiente", "declaraciones pendientes"))} en ${esc(d.titulo.toLowerCase())}${atrasadas.length ? ` · <span style="color:var(--rojo)">${esc(plural(atrasadas.length, "atrasada", "atrasadas"))}</span>` : ""}. Las fechas se corren al siguiente día hábil.</p></div>
        <div class="acciones">
          <div class="segmentado" role="group" aria-label="Tipo">${[["todos", "Todas"], ...Object.entries(d.tipos).map(([k, v]) => [k, v.nombre])].map(([v, t]) => `<button data-tipo="${v}" aria-pressed="${v === tipo}">${esc(t)}</button>`).join("")}</div>
        </div>
      </header>
      <section class="tarjeta" style="overflow:hidden">
        <div class="tarjeta-cab" style="padding-bottom:10px">
          <h2 class="tarjeta-titulo" style="font-size:19px">${esc(d.titulo)}</h2>
          <div class="acciones"><button class="boton icono" data-mes="${d.anterior}" aria-label="Mes anterior">${icono("izq", "s")}</button>
            <button class="boton pequeno" data-mes="${hoy.slice(0, 7)}">Hoy</button>
            <button class="boton icono" data-mes="${d.siguiente}" aria-label="Mes siguiente">${icono("der", "s")}</button></div>
        </div>
        <div class="calendario">${SEMANA.map((s) => `<div class="cal-dia-semana">${s}</div>`).join("")}${celdas}</div>
      </section>
      ${atrasadas.length ? bloque("Atrasadas", atrasadas, "Presentarlas cuanto antes") : ""}
      ${dia ? bloque(fechaLarga(dia).replace(/^./, (c) => c.toUpperCase()), seleccion, seleccion.length ? `${seleccion.filter((o) => o.estado === "presentada").length} de ${seleccion.length} presentadas` : "") : ""}`;

    el.querySelectorAll("[data-mes]").forEach((b) => b.onclick = () => irA(b.dataset.mes));
    el.querySelectorAll("[data-tipo]").forEach((b) => b.onclick = () => { tipo = b.dataset.tipo; pintar(); });
    el.querySelectorAll("[data-dia]").forEach((b) => b.onclick = () => {
      if (b.dataset.dia.slice(0, 7) !== d.mes) { irA(b.dataset.dia.slice(0, 7)).then(() => { dia = b.dataset.dia; pintar(); }); return; }
      dia = b.dataset.dia; pintar();
      el.querySelector(".tarjeta:last-of-type")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    el.querySelectorAll("[data-obligacion]").forEach((s) => s.onchange = async () => {
      try {
        await api.patch(`/api/obligaciones/${s.dataset.obligacion}`, { estado: s.value });
        aviso(s.value === "presentada" ? "Marcada como presentada" : "Estado actualizado");
        d = await api.get(`/api/vencimientos?mes=${d.mes}`);
        pintar();
      } catch (e) { aviso(e.message, { error: true }); }
    });
  }
  pintar();
}
