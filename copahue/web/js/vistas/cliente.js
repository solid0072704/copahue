// Ficha de cliente: resumen, declaraciones, documentos, honorarios y actividad.
import { api } from "../api.js";
import {
  COLOR_AGENTE, ESTADOS_OBL, NIVEL, anilloSalud, avatar, aviso, barras, cargando, clp, cuando, esc, estadoObligacion,
  etiquetaTipo, fechaCorta, hoja, icono, iconoAgente, monograma, relativa,
} from "../ui.js";
import { activarOpciones, formularioCliente, leerFormulario } from "./clientes.js";

const PESTANAS = [["resumen", "Resumen"], ["declaraciones", "Declaraciones"], ["documentos", "Documentos"], ["honorarios", "Honorarios"], ["actividad", "Actividad"]];
const MES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sept", "oct", "nov", "dic"];

export async function render(el, params, app) {
  const cid = Number(params[0]);
  let pestana = PESTANAS.some(([v]) => v === params[1]) ? params[1] : "resumen";
  el.innerHTML = cargando();
  let d = await api.get(`/api/clientes/${cid}`);
  const hoy = app.sesion.hoy;

  function pintar() {
    const c = d.cliente;
    el.innerHTML = `
      <a class="volver" href="#/clientes">${icono("izq", "s")}Clientes</a>
      <header class="ficha-cab">
        ${monograma(c.nombre_corto || c.razon_social, "l")}
        <div style="min-width:0">
          <h1 class="titulo">${esc(c.razon_social)}</h1>
          <div class="ficha-datos"><span>${esc(c.rut)}</span><span>${esc(c.regimen_nombre)}</span>${c.comuna ? `<span>${esc(c.comuna)}</span>` : ""}${c.trabajadores ? `<span>${c.trabajadores} trabajadores</span>` : ""}</div>
        </div>
        <div class="ficha-salud">
          <div class="ficha-salud-texto"><strong>${esc(NIVEL[d.salud.nivel])}</strong>${d.salud.puntaje != null ? `Salud ${d.salud.puntaje}/100` : ""}</div>
          ${anilloSalud(d.salud, true)}
          <button class="boton" data-editar>${icono("lapiz", "s")}Editar</button>
        </div>
      </header>
      <div class="segmentado pestanas-vista" role="tablist">${PESTANAS.map(([v, t]) => `<button role="tab" data-pestana="${v}" aria-pressed="${v === pestana}">${t}</button>`).join("")}</div>
      <div data-contenido></div>`;
    el.querySelector("[data-editar]").onclick = editar;
    el.querySelector(".segmentado").addEventListener("click", (e) => {
      const b = e.target.closest("[data-pestana]");
      if (!b) return;
      pestana = b.dataset.pestana;
      history.replaceState(null, "", `#/clientes/${cid}/${pestana}`);
      el.querySelectorAll("[data-pestana]").forEach((x) => x.setAttribute("aria-pressed", x === b));
      pintarPestana();
    });
    pintarPestana();
  }

  function pintarPestana() {
    const caja = el.querySelector("[data-contenido]");
    caja.innerHTML = ({ resumen, declaraciones, documentos, honorarios, actividad })[pestana]();
    caja.querySelectorAll("[data-obligacion]").forEach((s) => s.onchange = () => cambiar(`/api/obligaciones/${s.dataset.obligacion}`, { estado: s.value },
      s.value === "presentada" ? "Marcada como presentada" : "Estado actualizado"));
    caja.querySelectorAll("[data-documento]").forEach((b) => b.onclick = () => {
      const recibido = b.getAttribute("aria-checked") !== "true";
      b.setAttribute("aria-checked", recibido);
      cambiar(`/api/documentos/${b.dataset.documento}`, { estado: recibido ? "recibido" : "solicitado" }, recibido ? "Documento recibido" : "Documento pendiente");
    });
    caja.querySelectorAll("[data-pagar]").forEach((b) => b.onclick = () => cambiar(`/api/facturas/${b.dataset.pagar}`, { pagada: true }, "Pago registrado"));
  }

  async function cambiar(ruta, cuerpo, mensaje) {
    try {
      await api.patch(ruta, cuerpo);
      d = await api.get(`/api/clientes/${cid}`);
      aviso(mensaje);
      pintar();
    } catch (e) { aviso(e.message, { error: true }); }
  }

  function resumen() {
    const c = d.cliente;
    const ind = d.indicadores;
    const ultimo = ind[ind.length - 1];
    const previo = ind[ind.length - 2];
    const variacion = ultimo && previo && previo.ventas ? Math.round(((ultimo.ventas - previo.ventas) / previo.ventas) * 100) : null;
    const etiquetas = ind.map((i) => MES[Number(i.periodo.slice(5, 7)) - 1]);
    const servicios = c.servicios.map((s) => `<span class="chip">${esc(d.servicios[s] || s)}</span>`).join("");
    return `<div class="rejilla dos">
      <div class="columna">
        ${ind.length ? `<section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">Ventas y compras netas</h2>
          <div class="leyenda"><span><i></i>Ventas</span><span><i class="b"></i>Compras</span></div></div>
          <div class="tarjeta-cuerpo">${barras([{ nombre: "Ventas", clase: "barra-a", valores: ind.map((i) => i.ventas) }, { nombre: "Compras", clase: "barra-b", valores: ind.map((i) => i.compras) }], etiquetas)}
          <div class="cifras">
            <div class="cifra"><div class="cifra-etiqueta">Ventas de ${esc(MES[Number(ultimo.periodo.slice(5, 7)) - 1])}</div><div class="cifra-valor">${esc(clp(ultimo.ventas))}</div>
              ${variacion != null ? `<div class="variacion ${variacion >= 0 ? "sube" : "baja"}">${variacion >= 0 ? "▲" : "▼"} ${Math.abs(variacion)}% vs. mes anterior</div>` : ""}</div>
            <div class="cifra"><div class="cifra-etiqueta">IVA del mes</div><div class="cifra-valor">${esc(clp(ultimo.iva_debito - ultimo.iva_credito))}</div>
              <div class="t3 pequeno">${ultimo.iva_debito >= ultimo.iva_credito ? "Débito menos crédito" : "Remanente de crédito"}</div></div>
            <div class="cifra"><div class="cifra-etiqueta">Remuneraciones</div><div class="cifra-valor">${esc(clp(ultimo.remuneraciones))}</div>
              <div class="t3 pequeno">${ultimo.trabajadores} trabajadores</div></div>
          </div></div></section>` : ""}
        <section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">Pendiente de revisión</h2></div>
          ${d.propuestas.length ? `<ul class="lista">${d.propuestas.map((p) => `<li><a class="fila con-icono" href="#/revisar/${p.id}">
            <span class="icono-cuadro ${COLOR_AGENTE[p.agente] || ""}">${iconoAgente(p.agente_icono)}</span>
            <span class="fila-cuerpo"><div class="fila-titulo truncar">${esc(p.titulo)}</div><div class="fila-sub truncar">${esc(p.agente_nombre)} · ${esc(relativa(p.creada))}</div></span>
            <span class="fila-fin">${icono("der", "xs")}</span></a></li>`).join("")}</ul>` : `<div class="vacio pequeno">Nada pendiente para este cliente.</div>`}
        </section>
      </div>
      <div class="columna">
        <section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">Contacto</h2></div><div class="tarjeta-cuerpo">
          <div class="contacto-linea">${icono("persona", "s")}${esc(c.contacto_nombre || "Sin contacto")}</div>
          ${c.contacto_email ? `<div class="contacto-linea">${icono("sobre", "s")}<a href="mailto:${esc(c.contacto_email)}">${esc(c.contacto_email)}</a></div>` : ""}
          ${c.contacto_telefono ? `<div class="contacto-linea">${icono("telefono", "s")}<a href="tel:${esc(c.contacto_telefono.replace(/\s/g, ""))}">${esc(c.contacto_telefono)}</a></div>` : ""}
        </div></section>
        <section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">Servicio</h2></div><div class="tarjeta-cuerpo">
          <div class="chips">${servicios}</div>
          <div class="contacto-linea" style="margin-top:12px">${icono("tarjeta", "s")}${esc(clp(c.honorario))} mensuales</div>
          ${c.responsable ? `<div class="contacto-linea">${avatar(c.responsable)}${esc(c.responsable.nombre)} <span class="t3">· responsable</span></div>` : ""}
          ${c.cliente_desde ? `<div class="contacto-linea">${icono("calendario", "s")}Cliente desde ${esc(fechaCorta(c.cliente_desde))} ${esc(c.cliente_desde.slice(0, 4))}</div>` : ""}
        </div></section>
        ${d.salud.motivos.length ? `<section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">Qué afecta su salud</h2></div>
          <div class="tarjeta-cuerpo"><ul class="lista">${d.salud.motivos.map((m) => `<li class="contacto-linea">${icono("alerta", "s")}${esc(m)}</li>`).join("")}</ul></div></section>` : ""}
        ${d.tareas.length ? `<section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">Tareas abiertas</h2></div>
          <ul class="lista compacta">${d.tareas.map((t) => `<li class="fila"><span class="fila-cuerpo"><div class="fila-titulo">${esc(t.titulo)}</div>
          <div class="fila-sub">${t.vence ? esc(cuando(t.vence, hoy)) : "Sin fecha"}</div></span></li>`).join("")}</ul></section>` : ""}
      </div></div>`;
  }

  function declaraciones() {
    if (!d.obligaciones.length) return `<section class="tarjeta"><div class="vacio pequeno">Este cliente no tiene declaraciones según sus servicios.</div></section>`;
    return `<section class="tarjeta" style="overflow:hidden"><table class="tabla"><thead><tr><th>Declaración</th><th>Período</th><th>Vence</th><th class="ocultar-movil">Monto</th><th>Estado</th></tr></thead><tbody>
      ${d.obligaciones.map((o) => `<tr>
        <td><span style="display:inline-flex;align-items:center;gap:10px">${etiquetaTipo(o)}<span class="t2 pequeno ocultar-movil">${esc(o.descripcion)}</span></span></td>
        <td>${esc(o.periodo_nombre)}</td>
        <td class="num">${esc(fechaCorta(o.vence))}${o.presentada_el ? `<div class="t3 pequeno">Presentada ${esc(fechaCorta(o.presentada_el))}</div>` : ""}</td>
        <td class="num ocultar-movil">${o.monto != null ? esc(clp(o.monto)) : `<span class="t3">—</span>`}</td>
        <td>${o.estado === "presentada" ? estadoObligacion(o) : `<span style="display:inline-flex;gap:8px;align-items:center">${o.atrasada ? estadoObligacion(o) : ""}
          <select class="selector" data-obligacion="${o.id}" aria-label="Estado">${Object.entries(ESTADOS_OBL).map(([v, [, t]]) => `<option value="${v}" ${v === o.estado ? "selected" : ""}>${t}</option>`).join("")}</select></span>`}</td>
      </tr>`).join("")}</tbody></table></section>`;
  }

  function documentos() {
    if (!d.documentos.length) return `<section class="tarjeta"><div class="vacio pequeno">No se piden documentos mensuales a este cliente.</div></section>`;
    const recibidos = d.documentos.filter((x) => x.estado === "recibido").length;
    return `<section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">Documentos de ${esc(d.periodo_nombre)}</h2><span class="t2 pequeno">${recibidos} de ${d.documentos.length} recibidos</span></div>
      <ul class="lista">${d.documentos.map((x) => `<li class="fila">
        <button class="casilla" role="checkbox" aria-checked="${x.estado === "recibido"}" aria-label="Recibido" data-documento="${x.id}">${icono("check")}</button>
        <span class="fila-cuerpo"><div class="fila-titulo">${esc(x.tipo)}</div><div class="fila-sub">${x.estado === "recibido" ? `Recibido el ${esc(fechaCorta(x.recibido_el))}`
          : x.solicitado_el ? `Solicitado el ${esc(fechaCorta(x.solicitado_el))}${x.recordatorios > 1 ? ` · ${x.recordatorios} avisos` : ""}` : "Aún no se solicita"}</div></span>
      </li>`).join("")}</ul></section>`;
  }

  function honorarios() {
    if (!d.facturas.length) return `<section class="tarjeta"><div class="vacio pequeno">Sin facturas emitidas.</div></section>`;
    return `<section class="tarjeta" style="overflow:hidden"><table class="tabla"><thead><tr><th>Factura</th><th>Período</th><th class="ocultar-movil">Vence</th><th class="der">Monto</th><th>Estado</th></tr></thead><tbody>
      ${d.facturas.map((f) => {
        const vencida = !f.pagada_el && f.vence < hoy;
        return `<tr><td class="num">N.º ${esc(f.folio)}</td><td>${esc(MES[Number(f.periodo.slice(5, 7)) - 1])} ${esc(f.periodo.slice(0, 4))}</td>
          <td class="num ocultar-movil">${esc(fechaCorta(f.vence))}</td><td class="der num">${esc(clp(f.monto))}</td>
          <td>${f.pagada_el ? `<span class="pill verde">Pagada ${esc(fechaCorta(f.pagada_el))}</span>` : `<span style="display:inline-flex;gap:8px;align-items:center">
            <span class="pill ${vencida ? "rojo" : ""}">${vencida ? `Vencida` : "Por vencer"}</span><button class="boton pequeno tinte" data-pagar="${f.id}">Registrar pago</button></span>`}</td></tr>`;
      }).join("")}</tbody></table></section>`;
  }

  function actividad() {
    return `<div class="rejilla mitad">
      <section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">Actividad</h2></div>
        ${d.actividad.length ? `<ul class="linea">${d.actividad.map((a) => `<li>${a.autor.tipo === "persona" ? avatar(a.autor) : `<span class="icono-mini">${iconoAgente(a.autor.icono)}</span>`}
          <div><div>${esc(a.texto)}</div><div class="cuando">${esc(a.autor.nombre)} · ${esc(relativa(a.fecha))}</div></div></li>`).join("")}</ul>` : `<div class="vacio pequeno">Sin actividad registrada.</div>`}
      </section>
      <section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">Correos enviados</h2></div>
        ${d.correos.length ? `<ul class="lista">${d.correos.map((m) => `<li class="fila"><span class="fila-cuerpo"><div class="fila-titulo truncar">${esc(m.asunto)}</div>
          <div class="fila-sub">${esc(relativa(m.fecha))} · ${m.estado === "enviado" ? "Enviado" : m.estado === "simulado" ? "Registrado (envío simulado)" : "Error"}</div></span></li>`).join("")}</ul>`
          : `<div class="vacio pequeno">Aún no hay correos para este cliente.</div>`}
      </section></div>`;
  }

  function editar() {
    hoja({
      titulo: "Editar cliente",
      aceptar: "Guardar",
      contenido: formularioCliente(d, d.cliente) + `
        <div class="bloque-form"><div class="grupo-titulo">Estado</div><div class="grupo">
          <div class="campo"><label for="f-estado">Estado</label><select id="f-estado" name="estado">${[["activo", "Activo"], ["incorporacion", "En incorporación"], ["pausado", "Pausado"]]
            .map(([v, t]) => `<option value="${v}" ${v === d.cliente.estado ? "selected" : ""}>${t}</option>`).join("")}</select></div>
          <div class="campo alto"><label for="f-notas">Notas</label><textarea id="f-notas" name="notas" placeholder="Visible solo para el estudio">${esc(d.cliente.notas || "")}</textarea></div>
        </div></div>`,
      alAbrir: activarOpciones,
      alAceptar: async (cuerpo) => {
        const valores = leerFormulario(cuerpo);
        try {
          await api.patch(`/api/clientes/${cid}`, valores);
          d = await api.get(`/api/clientes/${cid}`);
          app.olvidarClientes();
          aviso("Cambios guardados");
          pintar();
        } catch (e) { aviso(e.message, { error: true }); return false; }
      },
    });
  }

  pintar();
}
