import { api } from "../api.js";
import { NIVEL, anilloSalud, avatar, aviso, cargando, clp, cuando, esc, etiquetaTipo, hoja, icono, monograma } from "../ui.js";

const FILTROS = [["todos", "Todos"], ["ok", "Al día"], ["atencion", "Atención"], ["riesgo", "En riesgo"]];

export async function render(el, _params, app) {
  el.innerHTML = cargando();
  const datos = await api.get("/api/clientes");
  const hoy = app.sesion.hoy;
  let filtro = "todos";
  let texto = "";

  const conteo = (n) => n === "todos" ? datos.clientes.length : datos.clientes.filter((c) => c.salud.nivel === n).length;
  const honorarios = datos.clientes.filter((c) => c.estado === "activo").reduce((s, c) => s + c.honorario, 0);

  el.innerHTML = `
  <header class="encabezado">
    <div class="encabezado-texto">
      <h1 class="titulo">Clientes</h1>
      <p class="subtitulo">${datos.clientes.length} clientes · ${esc(clp(honorarios))} en honorarios mensuales</p>
    </div>
    <div class="acciones">
      <label class="campo-busqueda">${icono("buscar", "s")}<input type="search" placeholder="Buscar por nombre, RUT o giro" aria-label="Buscar clientes"></label>
      <button class="boton primario" data-nuevo>${icono("mas", "s")}Nuevo cliente</button>
    </div>
  </header>
  <div class="segmentado" role="group" aria-label="Filtrar por salud" style="margin-bottom:16px">
    ${FILTROS.map(([v, t]) => `<button data-filtro="${v}" aria-pressed="${v === filtro}">${t}<span>${conteo(v)}</span></button>`).join("")}
  </div>
  <section class="tarjeta" style="overflow:hidden"><table class="tabla">
    <thead><tr><th>Cliente</th><th class="ocultar-movil">Responsable</th><th>Salud</th><th class="ocultar-movil">Próximo vencimiento</th>
      <th class="ocultar-movil">Documentos</th><th class="der ocultar-movil">Honorario</th></tr></thead>
    <tbody></tbody></table></section>`;

  const cuerpo = el.querySelector("tbody");
  function pintar() {
    const norma = (t) => (t || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
    const q = norma(texto);
    const lista = datos.clientes.filter((c) => (filtro === "todos" || c.salud.nivel === filtro)
      && (!q || norma(`${c.razon_social} ${c.nombre_corto} ${c.rut} ${c.giro} ${c.comuna}`).includes(q)));
    cuerpo.innerHTML = lista.length ? lista.map((c) => `
      <tr data-href="#/clientes/${c.id}" tabindex="0">
        <td><div class="celda-cliente">${monograma(c.nombre_corto || c.razon_social)}<div class="truncar">
          <div class="nombre truncar">${esc(c.razon_social)}</div><div class="t2 pequeno truncar">${esc(c.rut)} · ${esc(c.giro)}</div></div></div></td>
        <td class="ocultar-movil">${c.responsable ? `<span style="display:inline-flex;align-items:center;gap:8px">${avatar(c.responsable)}<span class="truncar">${esc(c.responsable.nombre.split(" ")[0])}</span></span>` : "—"}</td>
        <td><span title="${esc(c.salud.motivos.join(" · ") || NIVEL[c.salud.nivel])}">${anilloSalud(c.salud)}</span></td>
        <td class="ocultar-movil">${c.proximo ? `<span style="display:inline-flex;align-items:center;gap:8px">${etiquetaTipo(c.proximo)}<span class="${c.proximo.atrasada ? "" : "t2"}" style="${c.proximo.atrasada ? "color:var(--rojo);font-weight:500" : ""}">${esc(cuando(c.proximo.vence, hoy))}</span></span>` : `<span class="t3">—</span>`}</td>
        <td class="ocultar-movil">${c.documentos_faltan ? `<span class="pill naranja">Faltan ${c.documentos_faltan}</span>` : `<span class="pill verde">Completos</span>`}</td>
        <td class="der num ocultar-movil">${esc(clp(c.honorario))}${c.deuda ? `<div class="pequeno" style="color:var(--rojo)">${esc(clp(c.deuda))} vencido</div>` : ""}</td>
      </tr>`).join("") : `<tr><td colspan="6"><div class="vacio pequeno">No hay clientes que coincidan.</div></td></tr>`;
  }

  el.querySelector("input[type=search]").addEventListener("input", (e) => { texto = e.target.value; pintar(); });
  el.querySelector(".segmentado").addEventListener("click", (e) => {
    const b = e.target.closest("[data-filtro]");
    if (!b) return;
    filtro = b.dataset.filtro;
    el.querySelectorAll("[data-filtro]").forEach((x) => x.setAttribute("aria-pressed", x === b));
    pintar();
  });
  cuerpo.addEventListener("click", (e) => { const tr = e.target.closest("tr[data-href]"); if (tr) location.hash = tr.dataset.href; });
  cuerpo.addEventListener("keydown", (e) => { const tr = e.target.closest("tr[data-href]"); if (tr && e.key === "Enter") location.hash = tr.dataset.href; });
  el.querySelector("[data-nuevo]").onclick = () => nuevoCliente(datos, app);
  pintar();
}

export function formularioCliente(datos, c = {}) {
  const servicios = c.servicios || ["contabilidad", "iva"];
  return `
    <div class="bloque-form"><div class="grupo-titulo">Empresa</div><div class="grupo">
      <div class="campo"><label for="f-razon">Razón social</label><input id="f-razon" name="razon_social" value="${esc(c.razon_social || "")}" placeholder="Obligatorio" required></div>
      <div class="campo"><label for="f-corto">Nombre corto</label><input id="f-corto" name="nombre_corto" value="${esc(c.nombre_corto || "")}" placeholder="Opcional"></div>
      ${c.id ? "" : `<div class="campo"><label for="f-rut">RUT</label><input id="f-rut" name="rut" value="" placeholder="76.123.456-7" inputmode="text"></div>`}
      <div class="campo"><label for="f-giro">Giro</label><input id="f-giro" name="giro" value="${esc(c.giro || "")}" placeholder="Actividad principal"></div>
      <div class="campo"><label for="f-comuna">Comuna</label><input id="f-comuna" name="comuna" value="${esc(c.comuna || "")}"></div>
      <div class="campo"><label for="f-regimen">Régimen</label><select id="f-regimen" name="regimen">${Object.entries(datos.regimenes).map(([k, v]) => `<option value="${k}" ${k === (c.regimen || "14D3") ? "selected" : ""}>${esc(v)}</option>`).join("")}</select></div>
      <div class="campo"><label for="f-trab">Trabajadores</label><input id="f-trab" name="trabajadores" type="number" min="0" value="${esc(c.trabajadores ?? 0)}"></div>
    </div></div>
    <div class="bloque-form"><div class="grupo-titulo">Servicios contratados</div><div class="grupo">
      <div class="campo"><span class="campo-etiqueta">Servicios</span><div class="opciones" data-servicios>${Object.entries(datos.servicios).map(([k, v]) => `<button type="button" class="opcion" data-servicio="${k}" aria-pressed="${servicios.includes(k)}">${esc(v)}</button>`).join("")}</div></div>
      <div class="campo"><label for="f-hon">Honorario mensual</label><input id="f-hon" name="honorario" type="number" min="0" step="1000" value="${esc(c.honorario ?? "")}" placeholder="$"></div>
      <div class="campo"><label for="f-resp">Responsable</label><select id="f-resp" name="responsable_id">${datos.equipo.map((u) => `<option value="${u.id}" ${u.id === c.responsable_id ? "selected" : ""}>${esc(u.nombre)}</option>`).join("")}</select></div>
    </div><div class="grupo-pie">Según los servicios, Tu Cartera arma el calendario de F29, Previred, LRE y F22, y la lista mensual de documentos.</div></div>
    <div class="bloque-form"><div class="grupo-titulo">Contacto</div><div class="grupo">
      <div class="campo"><label for="f-cnom">Nombre</label><input id="f-cnom" name="contacto_nombre" value="${esc(c.contacto_nombre || "")}"></div>
      <div class="campo"><label for="f-cmail">Correo</label><input id="f-cmail" name="contacto_email" type="email" value="${esc(c.contacto_email || "")}" placeholder="nombre@empresa.cl"></div>
      <div class="campo"><label for="f-ctel">Teléfono</label><input id="f-ctel" name="contacto_telefono" type="tel" value="${esc(c.contacto_telefono || "")}" placeholder="+56 9"></div>
    </div></div>`;
}

export function leerFormulario(cuerpo) {
  const valores = {};
  cuerpo.querySelectorAll("input[name], select[name], textarea[name]").forEach((i) => { valores[i.name] = i.value.trim(); });
  valores.servicios = [...cuerpo.querySelectorAll("[data-servicio][aria-pressed=true]")].map((b) => b.dataset.servicio);
  valores.trabajadores = Number(valores.trabajadores || 0);
  valores.honorario = Number(valores.honorario || 0);
  valores.responsable_id = Number(valores.responsable_id) || null;
  return valores;
}

export function activarOpciones(cuerpo) {
  cuerpo.querySelector("[data-servicios]").addEventListener("click", (e) => {
    const b = e.target.closest("[data-servicio]");
    if (b) b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") !== "true");
  });
}

function nuevoCliente(datos, app) {
  hoja({
    titulo: "Nuevo cliente",
    aceptar: "Agregar",
    contenido: formularioCliente(datos, { responsable_id: app.sesion.usuario.id }),
    alAbrir: activarOpciones,
    alAceptar: async (cuerpo) => {
      const valores = leerFormulario(cuerpo);
      if (valores.razon_social.length < 2) { aviso("Escribe la razón social.", { error: true }); return false; }
      try {
        const r = await api.post("/api/clientes", valores);
        app.olvidarClientes();
        aviso("Cliente agregado. Ya tiene su calendario tributario.");
        location.hash = `#/clientes/${r.id}`;
      } catch (e) { aviso(e.message, { error: true }); return false; }
    },
  });
}
