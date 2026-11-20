// Plan y uso: suscripción del estudio, cuota mensual de IA por persona y planes disponibles.
import { api } from "../api.js";
import { avatar, cargando, clp, esc, fechaLarga, icono, miles, plural, tokensCorto } from "../ui.js";

const ESTADO = {
  prueba: ["azul", "Prueba gratis"], activa: ["verde", "Activa"], vencida: ["rojo", "Inactiva"], cancelada: ["rojo", "Cancelada"],
};

export async function render(el, _params, app) {
  el.innerHTML = cargando();
  const d = await api.get("/api/plan");
  const s = d.suscripcion;
  const [claseEstado, textoEstado] = ESTADO[s.estado] || ["", s.estado];
  const pct = (usado, limite) => (limite ? Math.min(100, Math.round((usado * 100) / limite)) : 0);
  const color = (p) => (p >= 100 ? "rojo" : p >= 75 ? "naranja" : "");
  const correo = (asunto) => d.ventas_email
    ? `mailto:${encodeURIComponent(d.ventas_email)}?subject=${encodeURIComponent(asunto)}&body=${encodeURIComponent(`Estudio: ${app.sesion.estudio.nombre}\nUsuarios: ${d.usuarios}\n`)}`
    : "";
  const boton = (texto, asunto, clase = "") => {
    const href = correo(asunto);
    return href ? `<a class="boton ${clase}" href="${esc(href)}">${esc(texto)}</a>`
      : `<button class="boton ${clase}" disabled title="Falta configurar el correo de ventas (CUADRA_VENTAS_EMAIL)">${esc(texto)}</button>`;
  };

  const subtitulo = s.estado === "prueba" && s.vigente
    ? `Plan ${s.nombre} en prueba gratis hasta el ${fechaLarga(s.hasta)}.`
    : s.vigente ? `Plan ${s.nombre}${s.hasta ? `, pagado hasta el ${fechaLarga(s.hasta)}` : ""}.`
      : "Tu suscripción no está activa: las automatizaciones y la IA están en pausa. Tus datos siguen disponibles.";

  const filas = d.uso.map((u) => {
    const p = pct(u.usado, u.limite);
    const restantes = Math.max(0, Math.floor((u.limite - u.usado) / d.equivalencias.instruccion));
    return `<div class="medidor-fila">${avatar(u, "m")}
      <div style="flex:1;min-width:0">
        <div style="display:flex;justify-content:space-between;gap:10px"><span class="fila-titulo truncar">${esc(u.nombre)}${u.id === app.sesion.usuario.id ? ` <span class="t3">(tú)</span>` : ""}</span>
          <span class="t2 pequeno num">${esc(tokensCorto(u.usado))} de ${esc(tokensCorto(u.limite))}</span></div>
        <div class="barra-progreso ${color(p)}"><span style="width:${p}%"></span></div>
        <div class="fila-sub">${p >= 100 ? (d.bolsa ? "Cuota personal agotada · usando la bolsa de paquetes" : "Cuota agotada · sigue con plantillas y reglas")
          : `${p}% usado · ≈ ${plural(restantes, "pedido restante", "pedidos restantes")} en ⌘K`}</div>
      </div></div>`;
  }).join("");

  el.innerHTML = `
    <header class="encabezado"><div class="encabezado-texto"><h1 class="titulo">Plan y uso</h1><p class="subtitulo">${esc(subtitulo)}</p></div></header>
    <div class="rejilla dos">
      <section class="tarjeta">
        <div class="tarjeta-cab"><h2 class="tarjeta-titulo">Uso de IA en ${esc(d.periodo_nombre)}</h2>
          <span class="t2 pequeno num">${esc(tokensCorto(d.total))} de ${esc(tokensCorto(d.cupo_total))} tokens</span></div>
        <div style="padding:4px 18px 8px"><div class="barra-progreso ${color(pct(d.total, d.cupo_total))}" style="height:8px;border-radius:4px"><span style="width:${pct(d.total, d.cupo_total)}%"></span></div></div>
        ${filas}
        ${d.sin_responsable ? `<div class="medidor-fila"><span class="icono-cuadro gris">${icono("ciclo")}</span><div class="fila-cuerpo"><div class="fila-titulo">Automatizaciones sin responsable</div>
          <div class="fila-sub">${esc(tokensCorto(d.sin_responsable))} tokens en clientes sin responsable asignado</div></div></div>` : ""}
        ${d.extras ? `<div class="medidor-fila"><span class="icono-cuadro verde">${icono("mas")}</span><div class="fila-cuerpo"><div class="fila-titulo">Bolsa de paquetes</div>
          <div class="fila-sub">${esc(tokensCorto(d.bolsa))} disponibles de ${esc(tokensCorto(d.extras))} comprados este mes</div></div></div>` : ""}
        <div class="grupo-pie" style="padding:6px 18px 18px">Cada persona tiene su cuota mensual. Los pedidos en ⌘K usan la cuota de quien pide y los correos pulidos, la del responsable del cliente. Si se agota, Cuadra sigue funcionando con plantillas y reglas.${d.ia_activa ? "" : " En este estudio la IA está apagada, así que estos consumos son ilustrativos."}</div>
      </section>

      <div class="columna">
        <section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">Plan ${esc(s.nombre)}</h2><span class="pill ${claseEstado}">${esc(textoEstado)}</span></div>
          <div class="tarjeta-cuerpo">
            <div class="plan-precio">${esc(clp(d.total_mensual))}<small> al mes + IVA</small></div>
            <div class="t2 pequeno" style="margin-top:2px">${esc(plural(d.usuarios, "usuario", "usuarios"))} × ${esc(clp(s.precio_usuario))}</div>
            <ul class="lista" style="margin-top:14px">
              <li class="contacto-linea">${icono("personas", "s")}${d.usuarios} de ${s.max_usuarios ?? "∞"} usuarios</li>
              <li class="contacto-linea">${icono("edificio", "s")}${d.clientes} de ${s.max_clientes ?? "∞"} clientes</li>
              <li class="contacto-linea">${icono("medidor", "s")}${esc(miles(s.tokens_usuario))} tokens de IA por usuario al mes</li>
            </ul>
          </div></section>
        <section class="tarjeta"><div class="tarjeta-cab"><h2 class="tarjeta-titulo">¿Se acabó la cuota?</h2></div>
          <div class="tarjeta-cuerpo"><p class="t2" style="font-size:13.5px">Suma un paquete de <strong style="color:var(--text)">${esc(miles(d.paquete.tokens))} tokens</strong> por ${esc(clp(d.paquete.precio))} + IVA. Queda en una bolsa para todo el equipo hasta fin de mes.</p>
            <div style="margin-top:14px">${boton("Comprar paquete", `Paquete de ${miles(d.paquete.tokens)} tokens`, "tinte")}</div></div></section>
      </div>
    </div>

    <h2 class="tarjeta-titulo" style="font-size:19px;margin:34px 0 14px">Planes</h2>
    <div class="planes">${d.planes.map((p) => {
      const actual = p.clave === s.plan && s.vigente;
      return `<section class="tarjeta plan ${actual ? "actual" : ""}">
        <div><h3 style="font-size:17px">${esc(p.nombre)}</h3><p class="t2 pequeno" style="margin-top:3px">${esc(p.para)}</p></div>
        <div class="plan-precio">${esc(clp(p.precio_usuario))}<small> por usuario al mes + IVA</small></div>
        <ul>
          <li>${icono("check", "xs")}${p.max_usuarios ? `Hasta ${p.max_usuarios} usuarios` : "Usuarios ilimitados"}</li>
          <li>${icono("check", "xs")}${p.max_clientes ? `Hasta ${p.max_clientes} clientes` : "Clientes ilimitados"}</li>
          <li>${icono("check", "xs")}${esc(tokensCorto(p.tokens_usuario))} tokens de IA por usuario al mes (≈ ${Math.floor(p.tokens_usuario / d.equivalencias.instruccion)} pedidos en ⌘K)</li>
          <li>${icono("check", "xs")}Las seis automatizaciones y calendario tributario</li>
        </ul>
        ${actual && s.estado === "activa" ? `<button class="boton" disabled>Plan actual</button>` : boton(actual ? "Contratar este plan" : `Elegir ${p.nombre}`, `Plan ${p.nombre}`, actual || p.clave === "profesional" ? "primario" : "")}
      </section>`;
    }).join("")}</div>
    <p class="t3 pequeno" style="margin-top:16px">Precios en pesos chilenos, sin IVA. Se cobra mensualmente por usuario activo. La prueba gratis dura ${d.prueba_dias} días e incluye ${esc(miles(d.prueba_tokens))} tokens por usuario.</p>`;
}
