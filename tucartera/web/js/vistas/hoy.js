import { api } from "../api.js";
import {
  COLOR_AGENTE, anilloSalud, avatar, aviso, cargando, clp, clpCorto, cuando, esc, estadoObligacion, etiquetaTipo,
  icono, iconoAgente, monograma, plural, relativa,
} from "../ui.js";

export async function render(el, _params, app) {
  el.innerHTML = cargando();
  const d = await api.get("/api/hoy");
  const k = d.kpis;
  const nombre = app.sesion.usuario.nombre.split(" ")[0];
  const hoy = app.sesion.hoy;

  const resumen = [];
  if (k.para_revisar) resumen.push(`${plural(k.para_revisar, "propuesta espera", "propuestas esperan")} tu visto bueno`);
  if (k.vencen) resumen.push(`${plural(k.vencen, "declaración vence", "declaraciones vencen")} en los próximos días`);
  const frase = resumen.length ? resumen.join(" y ") + "." : "Todo al día. No hay nada urgente por ahora.";
  const pctDocs = k.documentos_total ? Math.round((k.documentos_recibidos / k.documentos_total) * 100) : 100;

  el.innerHTML = `
  <header class="encabezado">
    <div class="encabezado-texto">
      <div class="sobretitulo">${esc(d.fecha)}</div>
      <h1 class="titulo">${esc(d.saludo)}, ${esc(nombre)}</h1>
      <p class="subtitulo">${esc(frase.charAt(0).toUpperCase() + frase.slice(1))}</p>
    </div>
    <div class="acciones"><button class="boton" data-pedir>${icono("buscar", "s")}Buscar o pedir <kbd>⌘K</kbd></button></div>
  </header>

  <section class="kpis">
    <a class="tarjeta kpi" href="#/revisar">
      <div class="kpi-etiqueta">${icono("bandeja", "s")}Para revisar</div>
      <div class="kpi-valor">${k.para_revisar}</div>
      <div class="kpi-pie">Listas para aprobar o editar</div>
    </a>
    <a class="tarjeta kpi" href="#/vencimientos">
      <div class="kpi-etiqueta">${icono("calendario", "s")}Vencen en 5 días hábiles</div>
      <div class="kpi-valor">${k.vencen}</div>
      <div class="kpi-pie">${k.atrasadas ? `<span class="punto rojo"></span>${plural(k.atrasadas, "atrasada", "atrasadas")}` : `<span class="punto verde"></span>Nada atrasado`}</div>
    </a>
    <a class="tarjeta kpi" href="#/documentos">
      <div class="kpi-etiqueta">${icono("documento", "s")}Documentos de ${esc(d.periodo)}</div>
      <div class="kpi-valor">${k.documentos_recibidos}<small> / ${k.documentos_total}</small></div>
      <div class="kpi-pie" style="display:block"><div class="barra-progreso ${pctDocs === 100 ? "verde" : ""}" style="margin-top:9px"><span style="width:${pctDocs}%"></span></div></div>
    </a>
    <a class="tarjeta kpi" href="#/cobranza">
      <div class="kpi-etiqueta">${icono("tarjeta", "s")}Honorarios vencidos</div>
      <div class="kpi-valor">${esc(clpCorto(k.vencido))}</div>
      <div class="kpi-pie">${k.vencido ? `<span class="punto naranja"></span>` : `<span class="punto verde"></span>`}${esc(clp(k.por_cobrar))} por cobrar</div>
    </a>
  </section>

  <div class="rejilla dos">
    <div class="columna">
      <section class="tarjeta">
        <div class="tarjeta-cab"><h2 class="tarjeta-titulo">Para revisar</h2>${k.para_revisar ? `<a class="enlace-ver" href="#/revisar">Ver ${k.para_revisar}${icono("der", "xs")}</a>` : ""}</div>
        ${d.destacadas.length ? `<ul class="lista">${d.destacadas.map((p) => `
          <li><a class="fila con-icono" href="#/revisar/${p.id}">
            <span class="icono-cuadro ${COLOR_AGENTE[p.agente] || ""}">${iconoAgente(p.agente_icono)}</span>
            <span class="fila-cuerpo"><div class="fila-titulo truncar">${esc(p.titulo)}</div>
              <div class="fila-sub truncar">${esc(p.cliente ? p.cliente.nombre : p.resumen)}${p.cliente && p.resumen ? ` · ${esc(p.resumen.split(" · ").slice(1).join(" · "))}` : ""}</div></span>
            <span class="fila-fin">${p.prioridad === 1 ? `<span class="punto rojo" title="Prioridad alta"></span>` : ""}${icono("der", "xs")}</span>
          </a></li>`).join("")}</ul>` : `<div class="vacio pequeno">Nada pendiente. Las automatizaciones dejarán aquí lo próximo.</div>`}
      </section>

      <section class="tarjeta">
        <div class="tarjeta-cab"><h2 class="tarjeta-titulo">Agenda tributaria</h2><a class="enlace-ver" href="#/vencimientos">Calendario${icono("der", "xs")}</a></div>
        ${d.agenda.length ? d.agenda.map((dia) => `
          <div class="dia">
            <div class="dia-cab"><span class="dia-nombre ${dia.fecha === hoy ? "hoy" : ""}">${esc(dia.etiqueta)}</span><span class="dia-fecha">${esc(dia.detalle)}</span>
              <span class="dia-cuenta">${dia.listas ? `${dia.listas} de ${dia.total} listas` : plural(dia.total, "declaración", "declaraciones")}</span></div>
            <ul class="lista compacta">${dia.items.slice(0, dia.fecha === hoy ? 8 : 4).map((o) => `
              <li><a class="fila" href="#/clientes/${o.cliente_id}/declaraciones">${etiquetaTipo(o)}
                <span class="fila-cuerpo truncar"><span class="fila-titulo">${esc(o.nombre_corto || o.razon_social)}</span>
                <span class="t3 pequeno"> · ${esc(o.periodo_nombre)}${o.atrasada ? ` · venció ${esc(cuando(o.vence, hoy).toLowerCase())}` : ""}</span></span>
                ${estadoObligacion(o)}</a></li>`).join("")}
              ${dia.items.length > (dia.fecha === hoy ? 8 : 4) ? `<li><a class="fila t2 pequeno" href="#/vencimientos/${dia.fecha.slice(0, 7)}">y ${dia.items.length - (dia.fecha === hoy ? 8 : 4)} más…</a></li>` : ""}</ul>
          </div>`).join("") : `<div class="vacio pequeno">Sin vencimientos en las próximas semanas.</div>`}
      </section>
    </div>

    <div class="columna">
      <section class="tarjeta">
        <div class="tarjeta-cab"><h2 class="tarjeta-titulo">Requieren atención</h2><a class="enlace-ver" href="#/clientes">Cartera${icono("der", "xs")}</a></div>
        ${d.riesgo.length ? `<ul class="lista">${d.riesgo.map((r) => `
          <li><a class="fila" href="#/clientes/${r.cliente.id}">${monograma(r.cliente.nombre)}
            <span class="fila-cuerpo"><div class="fila-titulo truncar">${esc(r.cliente.nombre)}</div><div class="fila-sub">${esc(r.motivos[0] || "")}</div></span>
            ${anilloSalud(r)}</a></li>`).join("")}</ul>` : `<div class="vacio pequeno">Toda la cartera está al día.</div>`}
      </section>

      <section class="tarjeta">
        <div class="tarjeta-cab"><h2 class="tarjeta-titulo">Tus tareas</h2><a class="enlace-ver" href="#/tareas">Todas${icono("der", "xs")}</a></div>
        ${d.tareas.length ? `<ul class="lista compacta">${d.tareas.map((t) => `
          <li class="fila"><button class="casilla" role="checkbox" aria-checked="false" aria-label="Marcar como hecha" data-tarea="${t.id}">${icono("check")}</button>
            <span class="fila-cuerpo"><div class="fila-titulo">${esc(t.titulo)}</div>
            <div class="fila-sub">${esc(t.nombre_corto || t.razon_social || "Interna")}${t.vence ? ` · ${esc(cuando(t.vence, hoy))}` : ""}</div></span></li>`).join("")}</ul>`
          : `<div class="vacio pequeno">No tienes tareas abiertas.</div>`}
      </section>

      <section class="tarjeta">
        <div class="tarjeta-cab"><h2 class="tarjeta-titulo">Actividad reciente</h2><a class="enlace-ver" href="#/actividad">Todo${icono("der", "xs")}</a></div>
        <ul class="linea">${d.actividad.map((a) => `
          <li>${a.autor.tipo === "persona" ? avatar(a.autor) : `<span class="icono-mini">${iconoAgente(a.autor.icono)}</span>`}
            <div><div>${esc(a.texto)}</div><div class="cuando">${esc(a.autor.nombre)} · ${esc(relativa(a.fecha))}</div></div></li>`).join("")}</ul>
      </section>
    </div>
  </div>`;

  el.querySelector("[data-pedir]").onclick = () => app.abrirPaleta();
  el.querySelectorAll("[data-tarea]").forEach((b) => b.onclick = async () => {
    b.setAttribute("aria-checked", "true");
    try {
      await api.patch(`/api/tareas/${b.dataset.tarea}`, { estado: "hecha" });
      setTimeout(() => b.closest("li").remove(), 400);
      aviso("Tarea completada");
    } catch (e) { b.setAttribute("aria-checked", "false"); aviso(e.message, { error: true }); }
  });
}
