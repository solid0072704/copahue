// Cliente de la API: JSON, encabezado anti-CSRF y mensajes de error legibles.

export class ErrorApi extends Error {
  constructor(mensaje, estado) {
    super(mensaje);
    this.estado = estado;
  }
}

let alExpirar = () => {};
export const configurar = ({ sesionExpirada }) => { alExpirar = sesionExpirada; };

async function pedir(metodo, ruta, cuerpo) {
  let respuesta;
  try {
    respuesta = await fetch(ruta, {
      method: metodo,
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-Cuadra": "1" },
      body: cuerpo === undefined ? undefined : JSON.stringify(cuerpo),
    });
  } catch {
    throw new ErrorApi("Sin conexión. Revisa tu red e inténtalo de nuevo.", 0);
  }
  let datos = null;
  try { datos = await respuesta.json(); } catch { /* respuesta sin cuerpo */ }
  if (!respuesta.ok) {
    if (respuesta.status === 401 && !ruta.startsWith("/api/auth/")) alExpirar();
    let mensaje = datos?.detail;
    if (Array.isArray(mensaje)) mensaje = "Revisa los datos ingresados.";
    throw new ErrorApi(mensaje || "Algo salió mal. Inténtalo de nuevo.", respuesta.status);
  }
  return datos;
}

export const api = {
  get: (ruta) => pedir("GET", ruta),
  post: (ruta, cuerpo = {}) => pedir("POST", ruta, cuerpo),
  patch: (ruta, cuerpo = {}) => pedir("PATCH", ruta, cuerpo),
};
