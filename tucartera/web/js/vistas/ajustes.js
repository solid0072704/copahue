// Ajustes del estudio, equipo, conexiones y apariencia.
import { api } from "../api.js";
import { MARCAS, avatar, aviso, cargando, esc, hoja, icono } from "../ui.js";

const TEMAS = [["auto", "Automático"], ["light", "Claro"], ["dark", "Oscuro"]];

function temaActual() {
  try { return localStorage.getItem("tucartera-tema") || "auto"; } catch { return "auto"; }
}
export function aplicarTema(tema) {
  if (tema === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", tema);
  try { localStorage.setItem("tucartera-tema", tema); } catch { /* sin almacenamiento */ }
}

export async function render(el, _params, app) {
  el.innerHTML = cargando();
  const [d, { comentarios }] = await Promise.all([api.get("/api/ajustes"), api.get("/api/comentarios")]);
  const TIPOS = { falta: ["naranja", "Le falta"], error: ["rojo", "No funciona"], idea: ["azul", "Idea"] };
  const e = d.estudio;
  const socio = app.sesion.usuario.rol === "socio";
  const instalada = window.matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
  const esIOS = /iphone|ipad|ipod/i.test(navigator.userAgent);
  const campo = (id, etiqueta, nombre, valor, extra = "") => `<div class="campo"><label for="${id}">${etiqueta}</label><input id="${id}" name="${nombre}" value="${esc(valor || "")}" ${extra}></div>`;

  el.innerHTML = `
    <header class="encabezado"><div class="encabezado-texto"><h1 class="titulo">Ajustes</h1><p class="subtitulo">${esc(e.nombre)}</p></div></header>
    <div style="max-width:720px">
      ${e.demo ? `<div class="franja"><span class="icono-cuadro naranja">${icono("info")}</span><div class="fila-cuerpo"><div class="fila-titulo">Estás en un estudio de ejemplo</div>
        <div class="fila-sub">Los clientes y correos son ficticios: nada sale de Tu Cartera. Puedes volver a empezar cuando quieras.</div></div>
        <button class="boton" data-reiniciar>Restablecer datos</button></div>` : ""}

      <form class="bloque-form" data-estudio>
        <div class="grupo-titulo">Estudio</div>
        <div class="grupo">
          ${campo("a-nombre", "Nombre", "nombre", e.nombre, "required")}
          ${campo("a-rut", "RUT", "rut", e.rut, 'placeholder="76.123.456-7"')}
          ${campo("a-email", "Correo", "email", e.email, 'type="email" placeholder="contacto@estudio.cl"')}
          ${campo("a-tel", "Teléfono", "telefono", e.telefono, 'type="tel"')}
          ${campo("a-firma", "Firma de correos", "firma", e.firma, 'placeholder="Dirección · teléfono"')}
        </div>
        <div class="grupo-pie" style="display:flex;justify-content:space-between;align-items:center;gap:12px">
          <span>La firma va al final de cada correo, después del nombre del responsable.</span>
          <button class="boton primario pequeno">Guardar</button></div>
      </form>

      <div class="bloque-form">
        <div class="grupo-titulo">Inicio de sesión</div>
        <div class="grupo">
          <div class="campo"><span class="icono-cuadro gris">${icono("sobre")}</span>
            <span style="flex:1;font-size:14px">Correo y contraseña<div class="t2 pequeno">${esc(app.sesion.usuario.email.replace(/\+[^@]+@/, "@"))}${d.clave_definida ? "" : " · sin contraseña"}</div></span></div>
          ${Object.entries(d.proveedores).map(([clave, p]) => {
            const ident = d.identidades.find((i) => i.proveedor === clave);
            return `<div class="campo"><span class="icono-cuadro" style="background:var(--bg-elev);box-shadow:inset 0 0 0 1px var(--sep)">${MARCAS[clave]}</span>
              <span style="flex:1;font-size:14px">${esc(p.nombre)}<div class="t2 pequeno">${ident ? `Conectada como ${esc(ident.email)}` : "Ingresa con tu cuenta de " + esc(p.nombre)}</div></span>
              ${ident ? `<button class="boton pequeno" data-desconectar="${ident.id}" data-nombre="${esc(p.nombre)}">Desconectar</button>`
                : `<a class="boton pequeno tinte" href="/api/auth/${clave}/iniciar?modo=conectar">Conectar</a>`}</div>`;
          }).join("")}
        </div>
        <div class="grupo-pie">Solo usamos tu nombre y correo para que ingreses. Tu Cartera no lee tus correos ni tus archivos.</div>
      </div>

      <div class="bloque-form">
        <div class="grupo-titulo" style="display:flex;justify-content:space-between;align-items:center">Equipo
          ${socio ? `<button class="boton fantasma pequeno" data-invitar>${icono("mas", "xs")}Agregar persona</button>` : ""}</div>
        <div class="grupo"><ul class="lista">${d.equipo.map((u) => `<li class="fila con-icono">${avatar(u, "m")}
          <span class="fila-cuerpo"><div class="fila-titulo">${esc(u.nombre)}${u.id === app.sesion.usuario.id ? ` <span class="t3">(tú)</span>` : ""}</div>
          <div class="fila-sub">${esc(u.cargo || u.rol)} · ${esc(u.email.replace(/\+[^@]+@/, "@"))}</div></span>
          <span class="fila-fin">${u.clientes} cliente${u.clientes === 1 ? "" : "s"}</span></li>`).join("")}</ul></div>
      </div>

      <div class="bloque-form">
        <div class="grupo-titulo">Conexiones</div>
        <div class="grupo">
          <div class="campo"><span class="campo-etiqueta">Correo saliente</span><span style="flex:1;text-align:right" class="${d.integraciones.correo ? "" : "t2"}">${d.integraciones.correo ? esc(d.integraciones.remitente) : "Modo de prueba"}</span></div>
          <div class="campo"><span class="campo-etiqueta">Redacción asistida</span><span style="flex:1;text-align:right" class="t2">${d.integraciones.redaccion ? "Activa" : "Desactivada"}</span></div>
          <div class="campo"><span class="campo-etiqueta">Modelo de IA</span><span style="flex:1;text-align:right" class="t2">${esc(d.integraciones.modelo)} (fijo)</span></div>
        </div>
        <div class="grupo-pie">Se configuran en el servidor con SMTP_HOST, SMTP_USUARIO, SMTP_CLAVE y ANTHROPIC_API_KEY. El uso de IA de cada persona se ve en <a href="#/plan">Plan y uso</a>.</div>
      </div>

      <div class="bloque-form">
        <div class="grupo-titulo">Apariencia</div>
        <div class="grupo"><div class="campo"><span class="campo-etiqueta">Tema</span><div style="flex:1;display:flex;justify-content:flex-end">
          <div class="segmentado" role="group">${TEMAS.map(([v, t]) => `<button data-tema="${v}" aria-pressed="${v === temaActual()}">${t}</button>`).join("")}</div></div></div></div>
      </div>

      <div class="bloque-form">
        <div class="grupo-titulo" style="display:flex;justify-content:space-between;align-items:center">Sugerencias anotadas
          <span>${comentarios.length ? `<button class="boton fantasma pequeno" data-copiar>${icono("documento", "xs")}Copiar todas</button>` : ""}
          <button class="boton fantasma pequeno" data-nueva-sugerencia>${icono("mas", "xs")}Anotar</button></span></div>
        <div class="grupo">${comentarios.length ? `<ul class="lista">${comentarios.map((c) => `<li class="fila" style="align-items:flex-start">
          <span class="pill ${TIPOS[c.tipo]?.[0] || ""}" style="margin-top:1px">${TIPOS[c.tipo]?.[1] || "Idea"}</span>
          <span class="fila-cuerpo"><div style="font-size:14px;white-space:pre-wrap">${esc(c.texto)}</div>
          <div class="fila-sub">${esc(c.vista || "General")} · ${esc(c.autor || "")} · ${esc(c.fecha.slice(0, 16).replace("T", " "))}</div></span></li>`).join("")}</ul>`
          : `<div class="vacio pequeno">Mientras usas Tu Cartera, toca «Sugerir una mejora» en la barra lateral para anotar lo que le falta.</div>`}</div>
      </div>

      <div class="bloque-form">
        <div class="grupo-titulo">Usar como app</div>
        <div class="grupo"><div class="campo" style="padding-top:8px;padding-bottom:8px">
          <span class="icono-cuadro">${icono("descargar")}</span>
          <span style="flex:1;font-size:13.5px">${instalada ? "Tu Cartera ya está instalada en este dispositivo." : app.instalacion
            ? "Instala Tu Cartera en este equipo para abrirla como una app, con su propio ícono."
            : esIOS ? "En iPhone o iPad: toca Compartir y luego «Agregar a inicio»."
              : "Desde el menú del navegador elige «Instalar Tu Cartera» o «Agregar a la pantalla de inicio»."}</span>
          ${app.instalacion && !instalada ? `<button class="boton primario pequeno" data-instalar>Instalar</button>` : ""}
        </div></div>
        <div class="grupo-pie">Funciona en computador, tablet y celular con la misma cuenta.</div>
      </div>

      <div class="bloque-form"><div class="grupo"><button class="fila" data-salir style="color:var(--rojo);justify-content:center;min-height:46px">${icono("salir", "s")}Cerrar sesión</button></div></div>
    </div>`;

  el.querySelector("[data-estudio]").onsubmit = async (ev) => {
    ev.preventDefault();
    const datos = Object.fromEntries(new FormData(ev.target));
    if ((datos.nombre || "").trim().length < 2) { aviso("Escribe el nombre del estudio.", { error: true }); return; }
    try {
      await api.patch("/api/ajustes", datos);
      await app.refrescar();
      document.querySelector(".marca-estudio").textContent = datos.nombre;
      aviso("Datos del estudio guardados");
    } catch (err) { aviso(err.message, { error: true }); }
  };
  el.querySelectorAll("[data-tema]").forEach((b) => b.onclick = () => {
    aplicarTema(b.dataset.tema);
    el.querySelectorAll("[data-tema]").forEach((x) => x.setAttribute("aria-pressed", x === b));
  });
  el.querySelector("[data-instalar]")?.addEventListener("click", async () => {
    app.instalacion.prompt();
    await app.instalacion.userChoice;
    app.instalacion = null;
    render(el, _params, app);
  });
  el.querySelectorAll("[data-desconectar]").forEach((b) => b.onclick = async () => {
    try {
      await api.post(`/api/cuenta/identidades/${b.dataset.desconectar}/desconectar`);
      aviso(`Cuenta de ${b.dataset.nombre} desconectada`);
      render(el, _params, app);
    } catch (err) { aviso(err.message, { error: true }); }
  });
  el.querySelector("[data-nueva-sugerencia]").onclick = () => app.sugerir();
  el.querySelector("[data-copiar]")?.addEventListener("click", async () => {
    const texto = comentarios.map((c) => `- [${TIPOS[c.tipo]?.[1] || "Idea"}] ${c.vista || "General"}: ${c.texto}`).join("\n");
    try { await navigator.clipboard.writeText(texto); aviso("Sugerencias copiadas. Pégalas en el chat para trabajarlas."); }
    catch { aviso("No se pudo copiar automáticamente.", { error: true }); }
  });
  el.querySelector("[data-salir]").onclick = async () => { await api.post("/api/auth/salir"); location.hash = ""; location.reload(); };
  el.querySelector("[data-reiniciar]")?.addEventListener("click", async (ev) => {
    ev.currentTarget.disabled = true;
    await api.post("/api/auth/demo");
    location.hash = "#/hoy";
    location.reload();
  });
  el.querySelector("[data-invitar]")?.addEventListener("click", () => hoja({
    titulo: "Agregar persona",
    aceptar: "Agregar",
    contenido: `<div class="grupo">
      <div class="campo"><label for="i-nombre">Nombre</label><input id="i-nombre" name="nombre" required></div>
      <div class="campo"><label for="i-email">Correo</label><input id="i-email" name="email" type="email" required></div>
      <div class="campo"><label for="i-cargo">Cargo</label><input id="i-cargo" name="cargo" placeholder="Contador, analista…"></div>
      <div class="campo"><label for="i-clave">Contraseña inicial</label><input id="i-clave" name="clave" type="password" minlength="8" placeholder="Mínimo 8 caracteres"></div>
    </div><div class="grupo-pie">Comparte la contraseña inicial con la persona por un canal seguro.</div>`,
    alAceptar: async (cuerpo) => {
      const datos = Object.fromEntries([...cuerpo.querySelectorAll("input")].map((i) => [i.name, i.value.trim()]));
      try {
        await api.post("/api/equipo", datos);
        aviso("Persona agregada al equipo");
        render(el, _params, app);
      } catch (err) { aviso(err.message, { error: true }); return false; }
    },
  }));
}
