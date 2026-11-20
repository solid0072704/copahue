// Registro de lo que hizo el equipo y las automatizaciones, agrupado por día.
import { api } from "../api.js";
import { avatar, cargando, esc, fechaLarga, iconoAgente, isoDia } from "../ui.js";

export async function render(el, _params, app) {
  el.innerHTML = cargando();
  const { actividad } = await api.get("/api/actividad");
  const dias = {};
  for (const a of actividad) (dias[isoDia(new Date(a.fecha))] ||= []).push(a);
  const hoy = app.sesion.hoy;
  const nombreDia = (iso) => iso === hoy ? "Hoy" : fechaLarga(iso).replace(/^./, (c) => c.toUpperCase());
  const hora = (f) => new Date(f).toLocaleTimeString("es-CL", { hour: "2-digit", minute: "2-digit" });

  el.innerHTML = `
    <header class="encabezado"><div class="encabezado-texto"><h1 class="titulo">Actividad</h1>
      <p class="subtitulo">Todo lo que se presentó, envió y asignó, por el equipo o por las automatizaciones.</p></div></header>
    ${Object.keys(dias).length ? Object.entries(dias).map(([dia, items]) => `
      <section class="tarjeta" style="margin-bottom:16px">
        <div class="tarjeta-cab"><h2 class="tarjeta-titulo">${esc(nombreDia(dia))}</h2><span class="t3 pequeno">${items.length}</span></div>
        <ul class="linea">${items.map((a) => `<li>
          ${a.autor.tipo === "persona" ? avatar(a.autor) : `<span class="icono-mini">${iconoAgente(a.autor.icono)}</span>`}
          <div style="flex:1;min-width:0"><div>${esc(a.texto)}</div>
            <div class="cuando">${esc(a.autor.nombre)} · ${esc(hora(a.fecha))}${a.cliente_id ? ` · <a href="#/clientes/${a.cliente_id}">${esc(a.nombre_corto || a.razon_social)}</a>` : ""}</div></div>
        </li>`).join("")}</ul>
      </section>`).join("") : `<section class="tarjeta"><div class="vacio pequeno">Aún no hay actividad.</div></section>`}`;
}
