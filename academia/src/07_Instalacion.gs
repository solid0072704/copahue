/**
 * TPDV Academy · Instalación y mantenimiento (se ejecutan desde el editor de Apps Script).
 */

/**
 * 1) Crea la planilla «TPDV Academy · Base de datos» con todas sus hojas.
 * 2) Carga el programa (módulos, lecciones y preguntas) desde 10_Catalogo.gs si las hojas están vacías.
 * 3) Revisa la carpeta de Drive y deja los archivos nuevos en «Por clasificar».
 * Se puede ejecutar varias veces: nunca borra ni sobrescribe lo que ya existe.
 */
function instalar() {
  validarEjecucionAdministrativa_();
  var registro = [];
  var dueno = correoDuenoScript_();
  if (!esCorreoPermitido_(dueno)) {
    registro.push('⚠️ La cuenta dueña (' + (dueno || 'desconocida') + ') no es @' + DOMINIO_PERMITIDO +
      '. Google solo entrega el correo de los visitantes cuando el script pertenece al mismo dominio: crea el proyecto con una cuenta @' +
      DOMINIO_PERMITIDO + '.');
  }

  var props = PropertiesService.getScriptProperties();
  var id = props.getProperty(PROPIEDAD_DB);
  var libro = null;
  if (id) {
    try { libro = SpreadsheetApp.openById(id); } catch (e) { libro = null; }
  }
  if (!libro) {
    libro = SpreadsheetApp.create('TPDV Academy · Base de datos');
    props.setProperty(PROPIEDAD_DB, libro.getId());
    registro.push('Planilla creada: ' + libro.getUrl());
  } else {
    registro.push('Planilla existente: ' + libro.getUrl());
  }
  reiniciarCaches_();

  Object.keys(HOJAS).forEach(function (clave) { hoja_(HOJAS[clave]); });
  ['Hoja 1', 'Sheet1', 'Hoja1'].forEach(function (nombre) {
    var sobrante = libro.getSheetByName(nombre);
    if (sobrante && libro.getSheets().length > 1) libro.deleteSheet(sobrante);
  });

  // Config: agrega solo las claves que falten.
  var existentes = {};
  leerTabla_(HOJAS.CONFIG).forEach(function (f) { existentes[aTexto_(f['Clave'])] = true; });
  var nuevasClaves = CONFIG_DEFECTO.filter(function (f) { return !existentes[f[0]]; }).map(function (f) {
    return { 'Clave': f[0], 'Valor': f[1], 'Descripción': f[2] };
  });
  anexarFilas_(HOJAS.CONFIG, nuevasClaves);
  if (nuevasClaves.length) registro.push('Config: ' + nuevasClaves.length + ' claves agregadas.');
  reiniciarCaches_();

  registro = registro.concat(cargarCatalogoSiVacio_());
  darFormato_(libro);

  try {
    var sync = sincronizarDesdeDrive_();
    registro.push('Drive revisado: ' + sync.revisados + ' archivos, ' + sync.nuevos + ' nuevos en «Por clasificar».');
  } catch (e) {
    registro.push('⚠️ No se pudo revisar la carpeta de Drive (' + e.message + '). Revisa CARPETA_CONTENIDO_ID y los permisos.');
  }
  invalidarEstructura_();

  registro.push('Siguiente paso: Implementar → Nueva implementación → Aplicación web → Ejecutar como «Yo» y acceso «Cualquier usuario de ' +
    DOMINIO_PERMITIDO + '». Luego ejecuta verificarPermisos desde el panel de administración.');
  var texto = registro.join('\n');
  Logger.log(texto);
  return texto;
}

/** Carga módulos, lecciones, preguntas y archivos excluidos del catálogo, solo en hojas vacías. */
function cargarCatalogoSiVacio_() {
  var mensajes = [];
  if (typeof catalogo_ !== 'function') return ['Sin catálogo inicial (10_Catalogo.gs).'];
  var catalogo = catalogo_();
  var cargas = [
    { def: HOJAS.MODULOS, filas: catalogo.modulos, nombre: 'módulos' },
    { def: HOJAS.LECCIONES, filas: catalogo.lecciones, nombre: 'lecciones' },
    { def: HOJAS.PREGUNTAS, filas: catalogo.preguntas, nombre: 'preguntas' },
    { def: HOJAS.POR_CLASIFICAR, filas: catalogo.excluidos, nombre: 'archivos excluidos (revisados, no publicados)' }
  ];
  cargas.forEach(function (c) {
    if (leerTabla_(c.def).length || !c.filas || !c.filas.length) return;
    anexarFilas_(c.def, c.filas);
    mensajes.push('Cargados ' + c.filas.length + ' ' + c.nombre + '.');
  });
  return mensajes;
}

function darFormato_(libro) {
  try {
    Object.keys(HOJAS).forEach(function (clave) {
      var def = HOJAS[clave];
      var hoja = libro.getSheetByName(def.nombre);
      if (!hoja) return;
      hoja.getRange(1, 1, 1, def.columnas.length).setFontWeight('bold').setBackground('#005A82').setFontColor('#FFFFFF');
      hoja.setFrozenRows(1);
    });
    var siNo = SpreadsheetApp.newDataValidation().requireValueInList(['SI', 'NO'], true).setAllowInvalid(true).build();
    [[HOJAS.MODULOS, 'Visible'], [HOJAS.LECCIONES, 'Obligatoria'], [HOJAS.LECCIONES, 'Visible'],
      [HOJAS.POR_CLASIFICAR, 'Incluir (SI/NO)']].forEach(function (par) {
      var hoja = libro.getSheetByName(par[0].nombre);
      var col = par[0].columnas.indexOf(par[1]) + 1;
      if (hoja && col > 0) hoja.getRange(2, col, Math.max(hoja.getMaxRows() - 1, 1), 1).setDataValidation(siNo);
    });
  } catch (e) {
    Logger.log('Formato omitido: ' + e.message);
  }
}

/** Revisa Drive todos los días entre 6 y 7 AM (hora de Chile). */
function activarSincronizacionDiaria() {
  validarEjecucionAdministrativa_();
  ScriptApp.getProjectTriggers().forEach(function (t) {
    if (t.getHandlerFunction() === 'sincronizacionProgramada') ScriptApp.deleteTrigger(t);
  });
  ScriptApp.newTrigger('sincronizacionProgramada').timeBased().everyDays(1).atHour(6).create();
  Logger.log('Sincronización diaria activada.');
  return 'Sincronización diaria activada.';
}

function sincronizacionProgramada() {
  validarEjecucionAdministrativa_();
  var resumen = sincronizarDesdeDrive_();
  Logger.log(JSON.stringify(resumen));
  return resumen;
}

/** Desde el editor o un activador corre como la cuenta dueña; desde el navegador exige administrador. */
function validarEjecucionAdministrativa_() {
  var activo = '';
  try { activo = String(Session.getActiveUser().getEmail() || '').toLowerCase(); } catch (e) { activo = ''; }
  var dueno = correoDuenoScript_();
  if (activo && activo !== dueno && !esAdmin_(activo)) {
    throw new Error('Esta acción es solo para administradores de la academia.');
  }
}
