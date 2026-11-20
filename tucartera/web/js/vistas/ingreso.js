import { api } from "../api.js";
import { LOGO, MARCAS, esc, icono } from "../ui.js";

// Mensaje que deja el servidor al volver de Google o Microsoft (se lee una sola vez).
let errorInicial = new URLSearchParams(location.search).get("error_acceso") || "";
if (errorInicial) history.replaceState(null, "", location.pathname + location.hash);

export function render(raiz, alEntrar) {
  let modo = "ingresar";
  const pintar = () => {
    const registro = modo === "registro";
    raiz.innerHTML = `<div class="ingreso"><div class="ingreso-caja">
      <div class="marca-logo">${LOGO}</div>
      <h1>${registro ? "Crea la cuenta de tu estudio" : "Tu Cartera"}</h1>
      <p class="subtitulo">${registro ? "Empieza con tu cartera en minutos. Puedes invitar a tu equipo después."
        : "La cartera de tu estudio contable, siempre al día. Vencimientos, documentos y cobranza que avanzan solos."}</p>
      <div class="proveedores">
        <a class="boton grande proveedor" href="/api/auth/google/iniciar">${MARCAS.google}Continuar con Google</a>
        <a class="boton grande proveedor" href="/api/auth/microsoft/iniciar">${MARCAS.microsoft}Continuar con Microsoft</a>
      </div>
      <div class="ingreso-separador">o con tu correo</div>
      <form novalidate>
        ${registro ? `<input class="entrada" name="estudio" placeholder="Nombre del estudio" autocomplete="organization" required>
        <input class="entrada" name="nombre" placeholder="Tu nombre" autocomplete="name" required>` : ""}
        <input class="entrada" name="email" type="email" placeholder="Correo electrónico" autocomplete="email" required>
        <input class="entrada" name="clave" type="password" placeholder="${registro ? "Contraseña (mínimo 8 caracteres)" : "Contraseña"}" autocomplete="${registro ? "new-password" : "current-password"}" required>
        <div class="error-form" role="alert">${esc(errorInicial)}</div>
        <button class="boton primario grande" type="submit">${registro ? "Crear cuenta · 14 días gratis" : "Iniciar sesión"}</button>
      </form>
      <button class="boton grande fantasma" data-demo style="width:100%;margin-top:14px">Explorar con un estudio de ejemplo</button>
      <p class="ingreso-pie">${registro ? `¿Ya tienes cuenta? <a href="#" data-cambiar>Inicia sesión</a>` : `¿Primera vez? <a href="#" data-cambiar>Crea la cuenta de tu estudio</a>`}</p>
      ${registro ? "" : `<div class="ingreso-rasgos">
        <div><span class="icono-cuadro">${icono("calendario")}</span><span><strong>Calendario tributario propio</strong>F29, Previred, LRE y F22 de cada cliente, con días hábiles y feriados.</span></div>
        <div><span class="icono-cuadro verde">${icono("bandeja")}</span><span><strong>Todo listo para aprobar</strong>Solicitudes de documentos, cobranza y avisos redactados; tú das el visto bueno.</span></div>
        <div><span class="icono-cuadro naranja">${icono("pulso")}</span><span><strong>Salud de la cartera</strong>Quién se atrasa, quién no responde y dónde hay oportunidades.</span></div>
      </div>`}
    </div></div>`;
    const form = raiz.querySelector("form");
    const error = raiz.querySelector(".error-form");
    form.onsubmit = async (e) => {
      e.preventDefault();
      const datos = Object.fromEntries(new FormData(form));
      const boton = form.querySelector("button[type=submit]");
      error.textContent = "";
      if (!datos.email || !datos.clave) { error.textContent = "Completa tu correo y contraseña."; return; }
      if (registro && datos.clave.length < 8) { error.textContent = "La contraseña debe tener al menos 8 caracteres."; return; }
      boton.disabled = true;
      errorInicial = "";
      try {
        await api.post(registro ? "/api/auth/registro" : "/api/auth/ingresar", datos);
        alEntrar();
      } catch (err) {
        error.textContent = err.message;
        boton.disabled = false;
      }
    };
    raiz.querySelector("[data-cambiar]").onclick = (e) => { e.preventDefault(); modo = registro ? "ingresar" : "registro"; pintar(); };
    raiz.querySelector("[data-demo]").onclick = async (e) => {
      const b = e.currentTarget;
      b.disabled = true;
      b.innerHTML = `<span class="giro s"></span> Preparando el estudio de ejemplo…`;
      try {
        await api.post("/api/auth/demo");
        alEntrar();
      } catch (err) {
        error.textContent = err.message;
        b.disabled = false;
        b.textContent = "Explorar con un estudio de ejemplo";
      }
    };
    raiz.querySelector("input")?.focus();
  };
  pintar();
  document.title = "Tu Cartera";
}
