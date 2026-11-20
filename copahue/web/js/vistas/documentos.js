// Documentos del mes por cliente, con casillas al estilo de Recordatorios.
import { api } from "../api.js";
import { avatar, aviso, cargando, esc, fechaCorta, icono, monograma, plural } from "../ui.js";

export async function render(el, _params, app) {
  el.innerHTML = cargando();
  let d = await api.get("/api/documentos");
  let filtro = "pendientes";

  function pintar() {
    const total = d.grupos.reduce((s, g) => s + g.total, 0);
    const recibidos = d.grupos.reduce((s, g) => s + g.recibidos, 0);
    const completos = d.grupos.filter((g) => g.recibidos === g.total).length;
    const grupos = d.grupos.filter((g) => filtro === "todos" || (filtro === "pendientes" ? g.recibidos < g.total : g.recibidos === g.total));
    const pct = total ? Math.round((recibidos / total) * 100) : 100;
    el.innerHTML = `
      <header class="encabezado">
        <div class="encabezado-texto"><h1 class="titulo">Documentos de ${esc(d.periodo_nombre)}</h1>
          <p class="subtitulo">${recibidos} de ${total} recibidos · ${esc(plural(completos, "cliente completo", "clientes completos"))}. El F29 vence el ${esc(d.f29_vence_texto)}.</p></div>
        <div class="acciones"><button class="boton primario" data-pedir>${icono("enviar", "s")}Pedir lo que falta</button></div>
      </header>
      <div class="barra-progreso ${pct === 100 ? "verde" : ""}" style="margin:-8px 0 22px;height:5px"><span style="width:${pct}%"></span></div>
      <div class="segmentado" style="margin-bottom:16px" role="group">
        ${[["pendientes", "Con pendientes"], ["completos", "Completos"], ["todos", "Todos"]].map(([v, t]) => `<button data-filtro="${v}" aria-pressed="${v === filtro}">${t}</button>`).join("")}
      </div>
      ${grupos.length ? `<div class="rejilla mitad">${grupos.map((g) => `
        <section class="tarjeta">
          <div class="tarjeta-cab" style="padding-bottom:4px">
            <a href="#/clientes/${g.cliente.id}/documentos" style="display:flex;align-items:center;gap:12px;color:inherit;min-width:0;text-decoration:none">
              ${monograma(g.cliente.nombre)}<span class="truncar"><div class="tarjeta-titulo truncar">${esc(g.cliente.nombre)}</div>
              <div class="t2 pequeno truncar">${esc(g.cliente.contacto || "")}</div></span></a>
            <span style="display:flex;align-items:center;gap:10px;flex:none">${g.responsable ? avatar(g.responsable) : ""}
              <span class="pill ${g.recibidos === g.total ? "verde" : "naranja"}">${g.recibidos}/${g.total}</span></span>
          </div>
          <ul class="lista compacta">${g.documentos.map((x) => `<li class="fila">
            <button class="casilla" role="checkbox" aria-checked="${x.estado === "recibido"}" aria-label="Recibido: ${esc(x.tipo)}" data-documento="${x.id}">${icono("check")}</button>
            <span class="fila-cuerpo"><div class="fila-titulo">${esc(x.tipo)}</div>
              <div class="fila-sub">${x.estado === "recibido" ? `Recibido el ${esc(fechaCorta(x.recibido_el))}` : x.solicitado_el
                ? `Pedido el ${esc(fechaCorta(x.solicitado_el))}${x.recordatorios > 1 ? ` · <span style="color:var(--naranja)">${x.recordatorios} avisos sin respuesta</span>` : ""}` : "Aún no se pide"}</div></span>
          </li>`).join("")}</ul>
        </section>`).join("")}</div>` : `<section class="tarjeta"><div class="vacio pequeno">${filtro === "pendientes" ? "Todos los clientes enviaron sus documentos." : "Nada que mostrar."}</div></section>`}`;

    el.querySelectorAll("[data-filtro]").forEach((b) => b.onclick = () => { filtro = b.dataset.filtro; pintar(); });
    el.querySelectorAll("[data-documento]").forEach((b) => b.onclick = async () => {
      const recibido = b.getAttribute("aria-checked") !== "true";
      b.setAttribute("aria-checked", recibido);
      try {
        await api.patch(`/api/documentos/${b.dataset.documento}`, { estado: recibido ? "recibido" : "solicitado" });
        d = await api.get("/api/documentos");
        setTimeout(pintar, recibido ? 350 : 0);
      } catch (e) { b.setAttribute("aria-checked", !recibido); aviso(e.message, { error: true }); }
    });
    el.querySelector("[data-pedir]").onclick = async (e) => {
      const b = e.currentTarget;
      b.disabled = true;
      try {
        const r = await api.post("/api/automatizaciones/ejecutar", { clave: "documentos" });
        await app.refrescar();
        const n = r.resultados[0];
        aviso(n.creadas ? `${plural(n.creadas, "solicitud lista", "solicitudes listas")} para revisar` : "Las solicitudes ya están en Para revisar",
          { enlace: { href: "#/revisar", texto: "Ver" } });
      } catch (err) { aviso(err.message, { error: true }); }
      b.disabled = false;
    };
  }
  pintar();
}
