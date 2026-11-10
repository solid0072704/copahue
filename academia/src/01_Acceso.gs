/**
 * TPDV Academy · Control de acceso.
 *
 * Hay dos barreras:
 *  1. La implementación web se publica con «Cualquier usuario de tuproyectodevida.cl»
 *     (appsscript.json → webapp.access = DOMAIN). Google bloquea a quien no es del dominio.
 *  2. Cada página y cada función que se llama desde el navegador vuelve a validar aquí
 *     que el correo termine exactamente en @tuproyectodevida.cl.
 */

/** Devuelve el usuario que está usando la aplicación. */
function usuarioActual_() {
  var email = '';
  try {
    email = String(Session.getActiveUser().getEmail() || '').trim().toLowerCase();
  } catch (e) {
    email = '';
  }
  var permitido = esCorreoPermitido_(email);
  return {
    email: email,
    permitido: permitido,
    admin: permitido && esAdmin_(email)
  };
}

/** true solo para cuentas nombre@tuproyectodevida.cl (sin subdominios ni variantes). */
function esCorreoPermitido_(email) {
  var texto = String(email || '').trim().toLowerCase();
  if (!/^[^\s@]+@[^\s@]+$/.test(texto)) return false;
  return texto.split('@')[1] === DOMINIO_PERMITIDO;
}

function correoDuenoScript_() {
  try {
    return String(Session.getEffectiveUser().getEmail() || '').trim().toLowerCase();
  } catch (e) {
    return '';
  }
}

/** Administradores: solo los correos de ADMINISTRADORES_ACADEMIA (11_Administradores.gs) que sean del dominio. */
function listaAdministradores_() {
  var lista = [];
  administradoresAcademia_().forEach(function (c) {
    var correo = String(c || '').trim().toLowerCase();
    if (esCorreoPermitido_(correo) && lista.indexOf(correo) === -1) lista.push(correo);
  });
  return lista;
}

function esAdmin_(email) {
  return esCorreoPermitido_(email) && listaAdministradores_().indexOf(String(email).toLowerCase()) !== -1;
}

/** Para funciones llamadas desde el navegador: corta la ejecución si el correo no es del dominio. */
function exigirUsuario_() {
  var usuario = usuarioActual_();
  if (!usuario.permitido) {
    throw new Error('Acceso restringido: ingresa con tu cuenta @' + DOMINIO_PERMITIDO + '.');
  }
  return usuario;
}

function exigirAdmin_() {
  var usuario = exigirUsuario_();
  if (!usuario.admin) throw new Error('Esta acción es solo para administradores de la academia.');
  return usuario;
}
