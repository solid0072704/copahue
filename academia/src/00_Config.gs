/**
 * TPDV Academy · Configuración base.
 *
 * Casi todo lo editable vive en la hoja «Config» de la base de datos (nombre, colores,
 * puntaje de aprobación, administradores…). El dominio permitido se fija SOLO aquí, a propósito:
 * nadie puede ampliar el acceso editando la planilla.
 */

/** Único dominio de correo que puede entrar a la academia. */
var DOMINIO_PERMITIDO = 'tuproyectodevida.cl';

/**
 * Administradores: las ÚNICAS cuentas que ven el panel con el avance de cada colaborador.
 * La lista vive en 11_Administradores.gs (ADMINISTRADORES_ACADEMIA). No se puede ampliar desde la planilla
 * y el dueño del script no es administrador si no está en la lista. Sin lista, nadie ve el panel.
 */
function administradoresAcademia_() {
  return typeof ADMINISTRADORES_ACADEMIA !== 'undefined' && Array.isArray(ADMINISTRADORES_ACADEMIA) ? ADMINISTRADORES_ACADEMIA : [];
}

/** Carpeta «TPDV ACADEMY» compartida en Drive (fuente del contenido). */
var CARPETA_ACADEMIA_ID = '1hCDqvmusCl0nnC8TvKChiH_9ibKsR_b1';

var VERSION_ACADEMIA = '1.0.0';

/**
 * Valores iniciales de la hoja «Config»: [clave, valor, descripción].
 * Los colores y tipografías corresponden al kit de marca; se cambian desde la planilla sin tocar código.
 */
var CONFIG_DEFECTO = [
  ['NOMBRE_ACADEMIA', 'TPDV Academy', 'Nombre que aparece en la cabecera y en el certificado.'],
  ['BAJADA', 'Programa de inducción para nuevos asesores comerciales', 'Frase que acompaña al nombre.'],
  ['PUNTAJE_APROBACION', '80', 'Porcentaje mínimo para aprobar cada evaluación (0 a 100).'],
  ['INTENTOS_MAXIMOS', '0', 'Intentos permitidos por evaluación. 0 = sin límite.'],
  ['ORDEN_OBLIGATORIO', 'SI', 'SI: cada módulo se desbloquea al completar el anterior. NO: ruta libre.'],
  ['CORREO_CONTACTO', '', 'Correo de ayuda que verán los colaboradores (opcional).'],
  ['CARPETA_CONTENIDO_ID', CARPETA_ACADEMIA_ID, 'Carpeta de Drive que se revisa al sincronizar contenido nuevo.'],
  // Kit de marca: Manual de Marca TPDV (dic-2025). Colores principales #005A82 · #007264 · #85B500 · #FFB829.
  ['COLOR_PRIMARIO', '#005A82', 'Kit de marca · azul TPDV: cabecera, botones y títulos.'],
  ['COLOR_SECUNDARIO', '#007264', 'Kit de marca · verde azulado TPDV: barras de avance y detalles.'],
  ['COLOR_ACENTO', '#FFB829', 'Kit de marca · amarillo TPDV: insignias y llamados a la acción.'],
  ['COLOR_EXITO', '#85B500', 'Kit de marca · verde TPDV: lecciones y módulos completados.'],
  ['COLOR_FONDO', '#F0F8FF', 'Color de fondo de la página.'],
  ['COLOR_TEXTO', '#1A1A1A', 'Color del texto principal.'],
  ['FUENTE_TITULOS', 'Montserrat', 'Kit de marca: tipografía de títulos (nombre exacto en Google Fonts).'],
  ['FUENTE_TEXTO', 'Lato', 'Kit de marca: tipografía del texto (nombre exacto en Google Fonts).'],
  ['LOGO_ARCHIVO_ID', '1XWLUOPw-nXc8ps8VOdk_sc7glwxIk_dD', 'Logo para fondo azul (cabecera): horizontal_blanco.png del set de logos 2026.'],
  ['LOGO_COLOR_ARCHIVO_ID', '1PpMp46Vcb0dZiZeGX9L1aglMz1APL_RM', 'Logo para fondo claro (certificado): Horizontal_multicolor.png del set 2026.'],
  ['LOGO_URL', '', 'URL pública de un logo alternativo (opcional; se usa si los IDs anteriores no se pueden leer).']
];

/** Lee la configuración combinando los valores por defecto con la hoja «Config». */
function config_() {
  if (config_.cache) return config_.cache;
  var valores = {};
  CONFIG_DEFECTO.forEach(function (fila) { valores[fila[0]] = fila[1]; });
  try {
    leerTabla_(HOJAS.CONFIG).forEach(function (fila) {
      var clave = String(fila['Clave'] || '').trim();
      if (clave) valores[clave] = String(fila['Valor'] === undefined || fila['Valor'] === null ? '' : fila['Valor']).trim();
    });
  } catch (e) {
    // Sin base de datos todavía: se usan los valores por defecto.
  }
  config_.cache = valores;
  return valores;
}

function configNumero_(clave, minimo, maximo) {
  var n = Number(config_()[clave]);
  if (!isFinite(n)) {
    var porDefecto = CONFIG_DEFECTO.filter(function (f) { return f[0] === clave; })[0];
    n = porDefecto ? Number(porDefecto[1]) : 0;
  }
  return Math.min(maximo, Math.max(minimo, n));
}

function configSi_(clave) {
  return esSi_(config_()[clave]);
}

/** Interpreta SI/SÍ/TRUE/X/1 como verdadero. */
function esSi_(valor) {
  if (valor === true) return true;
  var t = String(valor === undefined || valor === null ? '' : valor).trim().toUpperCase();
  return t === 'SI' || t === 'SÍ' || t === 'TRUE' || t === 'VERDADERO' || t === 'X' || t === '1' || t === 'S';
}
