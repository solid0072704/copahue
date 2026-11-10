/**
 * TPDV Academy · Base de datos en Google Sheets.
 *
 * Cada hoja se lee por el nombre de sus encabezados, así que se pueden reordenar columnas
 * o agregar columnas propias sin romper nada.
 */

var HOJAS = {
  CONFIG: { nombre: 'Config', columnas: ['Clave', 'Valor', 'Descripción'] },
  MODULOS: {
    nombre: 'Módulos',
    columnas: ['ID', 'Orden', 'Semana', 'Pilar', 'Título', 'Descripción', 'Objetivos', 'Tipo', 'Visible']
  },
  LECCIONES: {
    nombre: 'Lecciones',
    columnas: ['ID', 'ID módulo', 'Orden', 'Sección', 'Título', 'Tipo', 'ID archivo', 'URL', 'Descripción',
      'Duración (min)', 'Obligatoria', 'Visible', 'Origen']
  },
  PREGUNTAS: {
    nombre: 'Preguntas',
    columnas: ['ID módulo', 'N°', 'Pregunta', 'A', 'B', 'C', 'D', 'Correcta', 'Explicación', 'Fuente']
  },
  USUARIOS: {
    nombre: 'Usuarios',
    columnas: ['Email', 'Nombre', 'Área', 'Primer ingreso', 'Último ingreso', 'Lecciones completadas',
      'Avance %', 'Módulos completados', 'Certificado']
  },
  PROGRESO: { nombre: 'Progreso', columnas: ['Email', 'ID lección', 'ID módulo', 'Completada', 'Fecha'] },
  RESULTADOS: {
    nombre: 'Resultados',
    columnas: ['Fecha', 'Email', 'ID módulo', 'Tipo', 'Puntaje %', 'Aprobado', 'Intento', 'Evaluador', 'Comentario']
  },
  POR_CLASIFICAR: {
    nombre: 'Por clasificar',
    columnas: ['ID archivo', 'Título', 'Tipo', 'Carpeta', 'URL', 'Detectado', 'Asignar a módulo (ID)',
      'Incluir (SI/NO)', 'Nota']
  }
};

var PROPIEDAD_DB = 'ACADEMIA_DB_ID';

/** Limpia los cachés en memoria (cada ejecución de Apps Script parte limpia; se usa en pruebas). */
function reiniciarCaches_() {
  config_.cache = null;
  db_.cache = null;
  hoja_.cache = {};
}

function idBaseDatos_() {
  return PropertiesService.getScriptProperties().getProperty(PROPIEDAD_DB) || '';
}

/** Planilla de la academia. Falla con un mensaje claro si todavía no se ejecuta instalar(). */
function db_() {
  if (db_.cache) return db_.cache;
  var id = idBaseDatos_();
  if (!id) throw new Error('La academia aún no está instalada: ejecuta la función instalar() desde el editor.');
  db_.cache = SpreadsheetApp.openById(id);
  return db_.cache;
}

/** Devuelve la hoja; si no existe la crea con sus encabezados. */
function hoja_(def) {
  hoja_.cache = hoja_.cache || {};
  if (hoja_.cache[def.nombre]) return hoja_.cache[def.nombre];
  var libro = db_();
  var hoja = libro.getSheetByName(def.nombre);
  if (!hoja) {
    hoja = libro.insertSheet(def.nombre);
    hoja.getRange(1, 1, 1, def.columnas.length).setValues([def.columnas]);
    hoja.setFrozenRows(1);
    hoja.getRange(1, 1, 1, def.columnas.length).setFontWeight('bold');
  }
  hoja_.cache[def.nombre] = hoja;
  return hoja;
}

function encabezados_(hoja, def) {
  var ultima = Math.max(hoja.getLastColumn(), 1);
  var fila = hoja.getRange(1, 1, 1, ultima).getValues()[0].map(function (v) { return String(v).trim(); });
  // Si faltan columnas nuevas (actualización de versión), se agregan al final.
  var faltantes = def.columnas.filter(function (c) { return fila.indexOf(c) === -1; });
  if (faltantes.length) {
    var ultimoLleno = fila.length;
    while (ultimoLleno > 0 && fila[ultimoLleno - 1] === '') ultimoLleno--;
    hoja.getRange(1, ultimoLleno + 1, 1, faltantes.length).setValues([faltantes]);
    fila = fila.slice(0, ultimoLleno).concat(faltantes);
  }
  return fila;
}

/** Lee todas las filas como objetos {encabezado: valor}. Agrega _fila (número de fila en la hoja). */
function leerTabla_(def) {
  var hoja = hoja_(def);
  var cabecera = encabezados_(hoja, def);
  var ultimaFila = hoja.getLastRow();
  if (ultimaFila < 2) return [];
  var valores = hoja.getRange(2, 1, ultimaFila - 1, cabecera.length).getValues();
  var filas = [];
  valores.forEach(function (fila, i) {
    var vacia = fila.every(function (v) { return v === '' || v === null; });
    if (vacia) return;
    var obj = { _fila: i + 2 };
    cabecera.forEach(function (col, j) { if (col) obj[col] = fila[j]; });
    filas.push(obj);
  });
  return filas;
}

function filaDesdeObjeto_(cabecera, obj) {
  return cabecera.map(function (col) {
    var v = obj[col];
    return v === undefined || v === null ? '' : v;
  });
}

function anexarFilas_(def, objetos) {
  if (!objetos.length) return;
  var hoja = hoja_(def);
  var cabecera = encabezados_(hoja, def);
  var filas = objetos.map(function (o) { return filaDesdeObjeto_(cabecera, o); });
  hoja.getRange(hoja.getLastRow() + 1, 1, filas.length, cabecera.length).setValues(filas);
}

/** Actualiza solo las columnas indicadas de una fila existente. */
function actualizarFila_(def, numeroFila, cambios) {
  var hoja = hoja_(def);
  var cabecera = encabezados_(hoja, def);
  Object.keys(cambios).forEach(function (col) {
    var j = cabecera.indexOf(col);
    if (j !== -1) hoja.getRange(numeroFila, j + 1).setValue(cambios[col]);
  });
}

/** Ejecuta fn con un bloqueo del script para evitar escrituras cruzadas entre usuarios. */
function conBloqueo_(fn) {
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    return fn();
  } finally {
    lock.releaseLock();
  }
}

function aTexto_(v) {
  return v === undefined || v === null ? '' : String(v).trim();
}

function aFechaIso_(v) {
  if (!v) return '';
  var d = v instanceof Date ? v : new Date(v);
  return isNaN(d.getTime()) ? '' : d.toISOString();
}
