// Tareas del equipo, agrupadas por fecha al estilo de Recordatorios.
import { api } from "../api.js";
import { avatar, aviso, cargando, cuando, diasEntre, esc, icono } from "../ui.js";

export async function render(el, _params, app) {
  el.innerHTML = cargando();
  let d = await api.get("/api/tareas");
  const hoy = app.sesion.hoy;
  let quien = "mias";

  function grupoDe(t) {
    if (t.estado === "hecha") return "Completadas";
    if (!t.vence) return "Sin fecha";
    const n = diasEntre(hoy, t.vence);
    return n < 0 ? "Vencidas" : n === 0 ? "Hoy" : n <= 7 ? "Próximos 7 días" : "Más adelante";
  }

  function pintar() {
    const lista = d.tareas.filter((t) => quien === "todas" || t.responsable_id === app.sesion.usuario.id);
    const orden = ["Vencidas", "Hoy", "Próximos 7 días", "Más adelante", "Sin fecha", "Completadas"];
    const grupos = Object.fromEntries(orden.map((g) => [g, []]));
    lista.forEach((t) => grupos[grupoDe(t)].push(t));
    const abiertas = lista.filter((t) => t.estado === "abierta").length;
    el.innerHTML = `
      <header class="encabezado">
        <div class="encabezado-texto"><h1 class="titulo">Tareas</h1><p class="subtitulo">${abiertas} abierta${abiertas === 1 ? "" : "s"}. Las que aprobaste en Para revisar llegan aquí asignadas.</p></div>
        <div class="segmentado" role="group">${[["mias", "Mías"], ["todas", "Del equipo"]].map(([v, t]) => `<button data-quien="${v}" aria-pressed="${v === quien}">${t}</button>`).join("")}</div>
      </header>
      <form class="tarjeta" data-nueva style="display:flex;align-items:center;gap:12px;padding:10px 18px;margin-bottom:18px">
        <span class="casilla" aria-hidden="true" style="border-style:dashed"></span>
        <input name="titulo" placeholder="Nueva tarea" aria-label="Nueva tarea" style="flex:1;border:0;outline:none;background:none;font-size:14px;height:32px;min-width:0">
        <select class="selector" name="responsable_id" aria-label="Responsable">${d.equipo.map((u) => `<option value="${u.id}" ${u.id === app.sesion.usuario.id ? "selected" : ""}>${esc(u.nombre)}</option>`).join("")}</select>
        <input class="selector" name="vence" type="date" aria-label="Fecha" style="padding-right:8px">
        <button class="boton primario pequeno">Agregar</button>
      </form>
      ${orden.filter((g) => grupos[g].length).map((g) => `
        <section class="tarjeta" style="margin-bottom:16px">
          <div class="tarjeta-cab"><h2 class="tarjeta-titulo" style="${g === "Vencidas" ? "color:var(--rojo)" : ""}">${g}</h2><span class="t3 pequeno">${grupos[g].length}</span></div>
          <ul class="lista">${grupos[g].map((t) => `<li class="fila">
            <button class="casilla" role="checkbox" aria-checked="${t.estado === "hecha"}" aria-label="Completar" data-tarea="${t.id}">${icono("check")}</button>
            <span class="fila-cuerpo"><div class="fila-titulo" style="${t.estado === "hecha" ? "color:var(--text-3);text-decoration:line-through" : ""}">${esc(t.titulo)}</div>
              <div class="fila-sub">${[t.cliente_id ? `<a href="#/clientes/${t.cliente_id}">${esc(t.nombre_corto || t.razon_social)}</a>` : "",
                t.vence ? esc(cuando(t.vence, hoy)) : "", t.origen_nombre ? esc(t.origen_nombre) : ""].filter(Boolean).join(" · ")}</div>
              ${t.detalle && t.estado === "abierta" ? `<div class="fila-sub" style="margin-top:3px">${esc(t.detalle)}</div>` : ""}</span>
            ${t.responsable ? avatar(t.responsable) : ""}
          </li>`).join("")}</ul>
        </section>`).join("") || `<section class="tarjeta"><div class="vacio pequeno">No hay tareas.</div></section>`}`;

    el.querySelectorAll("[data-quien]").forEach((b) => b.onclick = () => { quien = b.dataset.quien; pintar(); });
    el.querySelectorAll("[data-tarea]").forEach((b) => b.onclick = async () => {
      const hecha = b.getAttribute("aria-checked") !== "true";
      b.setAttribute("aria-checked", hecha);
      try {
        await api.patch(`/api/tareas/${b.dataset.tarea}`, { estado: hecha ? "hecha" : "abierta" });
        d = await api.get("/api/tareas");
        setTimeout(pintar, 300);
      } catch (e) { b.setAttribute("aria-checked", !hecha); aviso(e.message, { error: true }); }
    });
    el.querySelector("[data-nueva]").onsubmit = async (e) => {
      e.preventDefault();
      const datos = Object.fromEntries(new FormData(e.target));
      if (datos.titulo.trim().length < 2) return;
      try {
        await api.post("/api/tareas", { titulo: datos.titulo, responsable_id: Number(datos.responsable_id), vence: datos.vence || null });
        d = await api.get("/api/tareas");
        aviso("Tarea agregada");
        pintar();
        el.querySelector("[data-nueva] input").focus();
      } catch (err) { aviso(err.message, { error: true }); }
    };
  }
  pintar();
}
