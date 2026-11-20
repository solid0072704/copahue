// Automatizaciones: qué hace cada una, si está activa y si pide aprobación o actúa sola.
import { api } from "../api.js";
import { COLOR_AGENTE, aviso, cargando, esc, icono, iconoAgente, plural, relativa } from "../ui.js";

export async function render(el, _params, app) {
  el.innerHTML = cargando();
  let d = await api.get("/api/automatizaciones");

  function pintar() {
    const integ = d.integraciones;
    const ia = d.ia || {};
    el.innerHTML = `
      ${d.vigente ? "" : `<div class="franja" style="background:var(--rojo-suave)"><span class="icono-cuadro rojo">${icono("alerta")}</span>
        <div class="fila-cuerpo"><div class="fila-titulo">Automatizaciones en pausa</div><div class="fila-sub">Tu suscripción no está activa. Vuelven a trabajar apenas renueves el plan.</div></div>
        <a class="boton primario" href="#/plan">Ver planes</a></div>`}
      <header class="encabezado">
        <div class="encabezado-texto"><h1 class="titulo">Automatizaciones</h1>
          <p class="subtitulo">Revisan la cartera en segundo plano y dejan cada correo y tarea lista para tu aprobación. Tú decides cuáles pueden actuar solas.</p></div>
        <div class="acciones"><button class="boton primario" data-todas>${icono("ciclo", "s")}Revisar todo ahora</button></div>
      </header>
      <div class="franja">
        <span class="icono-cuadro ${integ.correo ? "verde" : "naranja"}">${icono("sobre")}</span>
        <div class="fila-cuerpo"><div class="fila-titulo">Correo saliente ${integ.correo ? "conectado" : "en modo de prueba"}</div>
          <div class="fila-sub">${integ.correo ? "Los correos aprobados salen desde tu servidor SMTP." : "Los correos aprobados quedan registrados sin enviarse. Configura SMTP para enviarlos."}</div></div>
        <span class="icono-cuadro ${integ.redaccion ? "verde" : "gris"}">${icono("lapiz")}</span>
        <div class="fila-cuerpo"><div class="fila-titulo">Redacción e instrucciones ${integ.redaccion ? "activas" : "básicas"}</div>
          <div class="fila-sub">${integ.redaccion ? (ia.permitido ? `Borradores pulidos y pedidos abiertos en ⌘K. Llevas ${ia.porcentaje}% de tu cuota del mes.`
            : "Tu cuota de IA del mes se agotó: se usan plantillas y reglas hasta el próximo mes o hasta sumar un paquete.")
            : "Plantillas y pedidos frecuentes. Agrega una clave de Claude para pedidos abiertos."} <a href="#/plan">Plan y uso</a></div></div>
      </div>
      <div class="rejilla mitad">${d.automatizaciones.map((a) => `
        <section class="tarjeta automatizacion ${a.activo ? "" : "apagada"}">
          <div class="automatizacion-cab">
            <span class="icono-cuadro grande ${COLOR_AGENTE[a.clave] || ""}">${iconoAgente(a.icono)}</span>
            <div style="flex:1;min-width:0"><h3>${esc(a.nombre)}</h3><p>${esc(a.descripcion)}</p></div>
            <label class="interruptor" title="${a.activo ? "Desactivar" : "Activar"}"><input type="checkbox" data-activo="${a.clave}" ${a.activo ? "checked" : ""} aria-label="Activar ${esc(a.nombre)}"><span></span></label>
          </div>
          <p class="detalle-texto">${esc(a.detalle)}</p>
          <div class="segmentado" role="group" aria-label="Modo de ${esc(a.nombre)}" style="align-self:flex-start">
            <button data-modo="revisar" data-clave="${a.clave}" aria-pressed="${a.modo === "revisar"}">Pedir aprobación</button>
            <button data-modo="automatico" data-clave="${a.clave}" aria-pressed="${a.modo === "automatico"}">Actuar sola</button>
          </div>
          <div class="automatizacion-pie">
            <div class="estadisticas">
              <span><strong>${a.pendientes}</strong> por revisar</span>
              <span><strong>${a.aprobadas_30d}</strong> aprobadas en 30 días</span>
              <span>${a.ultima_ejecucion ? `${esc(relativa(a.ultima_ejecucion))} · ${esc(a.ultimo_resultado)}` : "Aún no se ejecuta"}</span>
            </div>
            <button class="boton pequeno" data-ejecutar="${a.clave}">Ejecutar</button>
          </div>
        </section>`).join("")}</div>
      <p class="t3 pequeno" style="margin-top:18px">${d.intervalo_min ? `Se ejecutan solas cada ${d.intervalo_min} minutos.` : "Se ejecutan cuando lo pides aquí o en ⌘K. Para que corran solas, define CUADRA_INTERVALO_MIN o programa «python -m cuadra --automatizaciones»."}</p>`;

    el.querySelectorAll("[data-activo]").forEach((i) => i.onchange = () => ajustar(i.dataset.activo, { activo: i.checked },
      i.checked ? "Automatización activada" : "Automatización desactivada"));
    el.querySelectorAll("[data-modo]").forEach((b) => b.onclick = () => {
      if (b.getAttribute("aria-pressed") === "true") return;
      ajustar(b.dataset.clave, { modo: b.dataset.modo }, b.dataset.modo === "automatico"
        ? "Actuará sola: enviará y asignará sin esperar aprobación" : "Pedirá tu aprobación antes de actuar");
    });
    el.querySelectorAll("[data-ejecutar]").forEach((b) => b.onclick = () => ejecutar(b, b.dataset.ejecutar));
    el.querySelector("[data-todas]").onclick = (e) => ejecutar(e.currentTarget, null);
  }

  async function ajustar(clave, cambios, mensaje) {
    try {
      await api.patch(`/api/automatizaciones/${clave}`, cambios);
      d = await api.get("/api/automatizaciones");
      aviso(mensaje);
      pintar();
    } catch (e) { aviso(e.message, { error: true }); }
  }

  async function ejecutar(boton, clave) {
    boton.disabled = true;
    const texto = boton.innerHTML;
    boton.innerHTML = `<span class="giro s"></span>Revisando…`;
    try {
      const r = await api.post("/api/automatizaciones/ejecutar", clave ? { clave } : {});
      const nuevas = r.resultados.reduce((s, x) => s + x.creadas, 0);
      const hechas = r.resultados.reduce((s, x) => s + x.ejecutadas, 0);
      await app.refrescar();
      d = await api.get("/api/automatizaciones");
      pintar();
      aviso(nuevas || hechas ? [nuevas ? `${plural(nuevas, "propuesta nueva", "propuestas nuevas")}` : "", hechas ? `${plural(hechas, "acción ejecutada", "acciones ejecutadas")}` : ""].filter(Boolean).join(" · ")
        : "Revisado. No hay nada nuevo.", nuevas ? { enlace: { href: "#/revisar", texto: "Ver" } } : {});
    } catch (e) {
      aviso(e.message, { error: true });
      boton.disabled = false;
      boton.innerHTML = texto;
    }
  }
  pintar();
}
