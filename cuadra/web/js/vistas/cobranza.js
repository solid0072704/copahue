// Honorarios del estudio: qué está vencido, qué se cobró y quién debe.
import { api } from "../api.js";
import { aviso, barras, cargando, clp, clpCorto, esc, fechaCorta, icono, monograma, plural } from "../ui.js";

export async function render(el, _params, app) {
  el.innerHTML = cargando();
  let d = await api.get("/api/cobranza");
  let filtro = "pendientes";

  function pintar() {
    const r = d.resumen;
    const vencidas = d.facturas.filter((f) => f.estado === "vencida");
    const porVencer = d.facturas.filter((f) => f.estado === "por_vencer").length;
    const lista = d.facturas.filter((f) => filtro === "todas" || (filtro === "pendientes" ? f.estado !== "pagada" : f.estado === "pagada"));
    el.innerHTML = `
      <header class="encabezado">
        <div class="encabezado-texto"><h1 class="titulo">Cobranza</h1>
          <p class="subtitulo">${vencidas.length ? `${esc(plural(vencidas.length, "factura vencida", "facturas vencidas"))} por ${esc(clp(r.vencido))}.` : "Ninguna factura vencida."} Los recordatorios se redactan con tono cordial y se envían solo con tu aprobación.</p></div>
        <div class="acciones"><button class="boton primario" data-recordar>${icono("sobre", "s")}Preparar recordatorios</button></div>
      </header>
      <section class="kpis">
        <div class="tarjeta kpi"><div class="kpi-etiqueta"><span class="punto rojo"></span>Vencido</div><div class="kpi-valor">${esc(clpCorto(r.vencido))}</div><div class="kpi-pie">${vencidas.length ? esc(plural(vencidas.length, "factura", "facturas")) : "Nada vencido"}</div></div>
        <div class="tarjeta kpi"><div class="kpi-etiqueta"><span class="punto naranja"></span>Por vencer</div><div class="kpi-valor">${esc(clpCorto(r.por_vencer))}</div><div class="kpi-pie">${porVencer ? esc(plural(porVencer, "factura", "facturas")) : "Nada por vencer"}</div></div>
        <div class="tarjeta kpi"><div class="kpi-etiqueta"><span class="punto verde"></span>Cobrado este mes</div><div class="kpi-valor">${esc(clpCorto(r.cobrado_mes))}</div><div class="kpi-pie">de ${esc(clp(r.emitido_mes))} emitido</div></div>
        <div class="tarjeta kpi"><div class="kpi-etiqueta"><span class="punto azul"></span>Cartera mensual</div><div class="kpi-valor">${esc(clpCorto(r.cartera_mensual))}</div><div class="kpi-pie">Honorarios de clientes activos</div></div>
      </section>
      <div class="rejilla dos">
        <section class="tarjeta" style="overflow:hidden">
          <div class="tarjeta-cab" style="padding-bottom:12px"><h2 class="tarjeta-titulo">Facturas</h2>
            <div class="segmentado" role="group">${[["pendientes", "Pendientes"], ["pagadas", "Pagadas"], ["todas", "Todas"]].map(([v, t]) => `<button data-filtro="${v}" aria-pressed="${v === filtro}">${t}</button>`).join("")}</div></div>
          ${lista.length ? `<table class="tabla"><thead><tr><th>Cliente</th><th class="ocultar-movil">Período</th><th class="der">Monto</th><th>Estado</th></tr></thead><tbody>
            ${lista.map((f) => `<tr>
              <td><a class="celda-cliente" href="#/clientes/${f.cliente_id}/honorarios" style="color:inherit;text-decoration:none">${monograma(f.nombre_corto || f.razon_social)}
                <div class="truncar"><div class="nombre truncar">${esc(f.nombre_corto || f.razon_social)}</div><div class="t2 pequeno">N.º ${esc(f.folio)} · vence ${esc(fechaCorta(f.vence))}</div></div></a></td>
              <td class="ocultar-movil">${esc(f.periodo_nombre)}</td>
              <td class="der num">${esc(clp(f.monto))}</td>
              <td>${f.estado === "pagada" ? `<span class="pill verde">Pagada ${esc(fechaCorta(f.pagada_el))}</span>` : `<span style="display:inline-flex;gap:8px;align-items:center;flex-wrap:wrap">
                <span class="pill ${f.estado === "vencida" ? (f.dias_atraso > 44 ? "rojo" : "naranja") : ""}">${f.estado === "vencida" ? `${f.dias_atraso} días` : "Por vencer"}</span>
                <button class="boton pequeno tinte" data-pagar="${f.id}">Registrar pago</button></span>`}</td></tr>`).join("")}
          </tbody></table>` : `<div class="vacio pequeno">Nada que mostrar.</div>`}
        </section>
        <section class="tarjeta">
          <div class="tarjeta-cab"><h2 class="tarjeta-titulo">Emitido y cobrado</h2><div class="leyenda"><span><i></i>Cobrado</span><span><i class="b"></i>Emitido</span></div></div>
          <div class="tarjeta-cuerpo">${barras([
            { nombre: "Emitido", clase: "barra-b", valores: d.historico.map((h) => h.emitido) },
            { nombre: "Cobrado", clase: "barra-a", valores: d.historico.map((h) => h.cobrado) },
          ], d.historico.map((h) => h.nombre), { alto: 230, ancho: 380 })}</div>
        </section>
      </div>`;
    el.querySelectorAll("[data-filtro]").forEach((b) => b.onclick = () => { filtro = b.dataset.filtro; pintar(); });
    el.querySelectorAll("[data-pagar]").forEach((b) => b.onclick = async () => {
      b.disabled = true;
      try {
        await api.patch(`/api/facturas/${b.dataset.pagar}`, { pagada: true });
        d = await api.get("/api/cobranza");
        aviso("Pago registrado");
        pintar();
      } catch (e) { aviso(e.message, { error: true }); b.disabled = false; }
    });
    el.querySelector("[data-recordar]").onclick = async (e) => {
      const b = e.currentTarget;
      b.disabled = true;
      try {
        const res = (await api.post("/api/automatizaciones/ejecutar", { clave: "cobranza" })).resultados[0];
        await app.refrescar();
        aviso(res.creadas ? `${plural(res.creadas, "recordatorio listo", "recordatorios listos")} para revisar` : "Los recordatorios ya están en Para revisar",
          { enlace: { href: "#/revisar", texto: "Ver" } });
      } catch (err) { aviso(err.message, { error: true }); }
      b.disabled = false;
    };
  }
  pintar();
}
