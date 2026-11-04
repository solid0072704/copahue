/**
 * Codigo.gs — Aplicativo de conciliación / clasificación de cartolas bancarias.
 *
 * Flujo de trabajo:
 *   1. Menú "Conciliación bancaria > Cargar cartolas…": se suben las cartolas
 *      del período (xls/xlsx/csv) y se elige la empresa/cuenta de cada una.
 *   2. El motor (Motor.gs) clasifica cada movimiento según la historia de FLUJO
 *      y lo deja en la hoja CONCILIACION como IDENTIFICADO / POR REVISAR /
 *      PENDIENTE (o YA EN FLUJO si el movimiento ya estaba registrado).
 *   3. El analista completa los PENDIENTES y revisa los POR REVISAR.
 *   4. "Traspasar aprobados a FLUJO" agrega los movimientos a la historia,
 *      con lo que el modelo aprende para el mes siguiente.
 */

var HOJA = {
  FLUJO: 'FLUJO',
  CONCILIACION: 'CONCILIACION',
  REGLAS: 'REGLAS',
  PAGOS: 'PAGOS POR HACER',
  CUENTAS: 'CUENTAS BANCARIAS',
  PLAN: 'BBDD',
  BASE: 'base flujo caja',
  CONFIG: 'CONFIG',
  CARTOLA: 'CARTOLA',
  FACTURAS: 'FACTURAS COMPRA'
};

/** Columnas de la hoja CONCILIACION (en este orden). */
var COLS = ['ESTADO', 'CONFIANZA', 'FUENTE', 'EMPRESA', 'Fecha', 'Detalle Movimiento', 'Cheque o Cargo',
  'Deposito o Abono', 'Saldo', 'Docto. Nro.', 'Trn', 'Caja', 'Sucursal', 'DETALLE 1', 'C.COSTO', 'GLOSA',
  'CUENTA CONTABLE', 'NOMBRE CUENTA', 'CENTRO COSTO', 'RUT', 'RAZON SOCIAL', 'APROBAR', 'EXPLICACIÓN', 'LOTE'];
var C = {};
COLS.forEach(function (n, i) { C[n] = i; });

/** Colores por estado (función para no depender del orden de carga de los archivos .gs). */
function colores_() {
  var c = {};
  c[ESTADO.IDENTIFICADO] = '#d9ead3';
  c[ESTADO.REVISAR] = '#fff2cc';
  c[ESTADO.PENDIENTE] = '#f4cccc';
  c[ESTADO.DUPLICADO] = '#d9d9d9';
  c[ESTADO.MANUAL] = '#cfe2f3';
  return c;
}

/* ================================================================== */
/* Menú                                                                */
/* ================================================================== */

function onOpen() {
  SpreadsheetApp.getUi().createMenu('🏦 Conciliación bancaria')
    .addItem('1. Cargar cartolas y facturas SII (archivos)…', 'abrirCargaCartolas')
    .addItem('1b. Procesar cartola pegada en hoja "' + HOJA.CARTOLA + '"', 'procesarHojaCartola')
    .addSeparator()
    .addItem('2. Ver solo pendientes / por revisar', 'filtrarPendientes')
    .addItem('    Ver todos', 'quitarFiltro')
    .addItem('    Crear regla desde la fila seleccionada', 'crearReglaDesdeSeleccion')
    .addSeparator()
    .addItem('3. Traspasar aprobados a FLUJO', 'traspasarAFlujo')
    .addSeparator()
    .addItem('4. Cruzar facturas SII con movimientos', 'cruzarFacturas')
    .addItem('    Ver facturas sin pago / por revisar', 'filtrarFacturasPendientes')
    .addSeparator()
    .addItem('Evaluar precisión del modelo (últimos 3 meses)', 'evaluarModelo')
    .addItem('Limpiar hoja CONCILIACION', 'limpiarConciliacion')
    .addItem('Configuración inicial (crear hojas)', 'configurar')
    .addToUi();
}

function abrirCargaCartolas() {
  configurarSiFalta_();
  var html = HtmlService.createHtmlOutputFromFile('Dialogo').setWidth(760).setHeight(560);
  SpreadsheetApp.getUi().showModalDialog(html, 'Cargar cartolas bancarias y Registro de Compras SII');
}

/* ================================================================== */
/* Configuración                                                       */
/* ================================================================== */

function configurar() {
  configurarSiFalta_();
  SpreadsheetApp.getUi().alert('Hojas listas: ' + [HOJA.CONCILIACION, HOJA.REGLAS, HOJA.CUENTAS, HOJA.CONFIG, HOJA.CARTOLA].join(', ') +
    '.\n\nRevise CONFIG para ajustar los umbrales de confianza.');
}

function configurarSiFalta_() {
  var ss = SpreadsheetApp.getActive();
  if (!ss.getSheetByName(HOJA.FLUJO)) throw new Error('No existe la hoja "' + HOJA.FLUJO + '" con la historia de movimientos.');

  var sh = ss.getSheetByName(HOJA.CONFIG);
  var params = [
    ['umbralIdentificado', 0.80, 'Confianza mínima para marcar IDENTIFICADO (y aprobar automáticamente).'],
    ['umbralRevisar', 0.55, 'Confianza mínima para POR REVISAR. Bajo esto queda PENDIENTE (manual).'],
    ['usarPagosPorHacer', 'SI', 'Cruzar cargos con la hoja PAGOS POR HACER (monto exacto + nombre).'],
    ['diasPagoVentana', 60, 'Días máximos entre la semana de pago programada y el cargo en el banco.'],
    ['pesoOtraEmpresa', 0.70, 'Peso (0-1) de ejemplos de otras empresas frente a los de la misma empresa.'],
    ['vecinos', 12, 'Cantidad de movimientos parecidos que votan la clasificación.'],
    ['umbralFactura', 0.85, 'Facturas SII: confianza mínima para dar el pago por CONCILIADO y aplicarlo a CONCILIACION.'],
    ['umbralFacturaRevisar', 0.50, 'Facturas SII: confianza mínima para proponer un pago POR REVISAR.'],
    ['diasAntes', 30, 'Facturas SII: días que el pago puede ser ANTERIOR a la fecha de la factura (anticipos).'],
    ['diasDespues', 180, 'Facturas SII: días que el pago puede ser POSTERIOR a la fecha de la factura.'],
    ['buscarOtrasEmpresas', 'SI', 'Facturas SII: si no aparece el pago en la empresa, buscarlo en las demás (queda POR REVISAR).'],
    ['cuentaProveedores', 2102001, 'Facturas SII: cuenta contable que se asigna al pago de facturas (Proveedores Nacionales).']
  ];
  if (!sh) {
    sh = ss.insertSheet(HOJA.CONFIG);
    sh.getRange(1, 1, 1, 3).setValues([['PARÁMETRO', 'VALOR', 'DESCRIPCIÓN']]);
    sh.getRange('A1:C1').setFontWeight('bold').setBackground('#cfe2f3');
    sh.setColumnWidth(3, 520);
  }
  var ya = sh.getLastRow() > 1 ? sh.getRange(2, 1, sh.getLastRow() - 1, 1).getValues().map(function (r) { return String(r[0]).trim(); }) : [];
  var faltan = params.filter(function (p) { return ya.indexOf(p[0]) < 0; });
  if (faltan.length) sh.getRange(sh.getLastRow() + 1, 1, faltan.length, 3).setValues(faltan);

  sh = ss.getSheetByName(HOJA.REGLAS);
  if (!sh) {
    sh = ss.insertSheet(HOJA.REGLAS);
    var enc = ['ACTIVA', 'EMPRESA (vacío = todas)', 'DETALLE CONTIENE (texto o /regex/)', 'TIPO (CARGO/ABONO)', 'MONTO MIN',
      'MONTO MAX', 'DETALLE 1', 'C.COSTO', 'GLOSA', 'CUENTA CONTABLE', 'NOMBRE CUENTA', 'CENTRO COSTO', 'RUT', 'RAZON SOCIAL'];
    sh.getRange(1, 1, 1, enc.length).setValues([enc]).setFontWeight('bold').setBackground('#cfe2f3').setWrap(true);
    sh.getRange(2, 1, 1, enc.length).setValues([[true, '', 'COM.MANTENCION', 'CARGO', '', '', '', 'General', 'Comisión mantención plan',
      2102001, 'Proveedores Nacionales', 'General', '', '']]);
    sh.getRange('A2:A200').insertCheckboxes();
    sh.getRange('D2:D200').setDataValidation(SpreadsheetApp.newDataValidation().requireValueInList(['CARGO', 'ABONO'], true).setAllowInvalid(true).build());
    sh.setFrozenRows(1);
    sh.getRange('A2').setNote('Las reglas tienen prioridad sobre el modelo. Se evalúan de arriba hacia abajo; aplica la primera que coincide.');
  }

  sh = ss.getSheetByName(HOJA.CUENTAS);
  if (!sh) {
    sh = ss.insertSheet(HOJA.CUENTAS);
    sh.getRange(1, 1, 1, 5).setValues([['N° CUENTA', 'EMPRESA (como en FLUJO)', 'TITULAR EN CARTOLA', 'BANCO', 'RUT TITULAR']])
      .setFontWeight('bold').setBackground('#cfe2f3');
    sh.setFrozenRows(1);
    sh.getRange('A1').setNote('Se completa sola cada vez que se confirma la empresa de una cartola. Puede editarse.\n' +
      'El RUT TITULAR se usa para saber en qué cuentas buscar el pago de las facturas del SII de cada empresa.');
  } else if (!sh.getRange(1, 5).getValue()) {
    sh.getRange(1, 5).setValue('RUT TITULAR').setFontWeight('bold').setBackground('#cfe2f3');
  }

  sh = ss.getSheetByName(HOJA.CARTOLA);
  if (!sh) {
    sh = ss.insertSheet(HOJA.CARTOLA);
    sh.getRange('A1').setValue('Pegue aquí la cartola tal cual la descarga del banco (incluyendo la línea de la cuenta y los encabezados) y use el menú 1b.');
  }

  hojaConciliacion_();
}

function leerConfig_() {
  var cfg = {};
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.CONFIG);
  if (sh && sh.getLastRow() > 1) {
    sh.getRange(2, 1, sh.getLastRow() - 1, 2).getValues().forEach(function (r) {
      if (r[0]) cfg[String(r[0]).trim()] = r[1];
    });
  }
  ['umbralIdentificado', 'umbralRevisar', 'diasPagoVentana', 'pesoOtraEmpresa', 'vecinos', 'umbralFactura', 'umbralFacturaRevisar',
   'diasAntes', 'diasDespues', 'cuentaProveedores'].forEach(function (k) {
    if (cfg[k] !== undefined && cfg[k] !== '') cfg[k] = mtNum(cfg[k]); else delete cfg[k];
  });
  cfg.usarPagosPorHacer = String(cfg.usarPagosPorHacer || 'SI').toUpperCase().indexOf('S') === 0;
  if (cfg.buscarOtrasEmpresas !== undefined && cfg.buscarOtrasEmpresas !== '')
    cfg.buscarOtrasEmpresas = String(cfg.buscarOtrasEmpresas).toUpperCase().indexOf('S') === 0;
  else delete cfg.buscarOtrasEmpresas;
  return cfg;
}

function hojaConciliacion_() {
  var ss = SpreadsheetApp.getActive();
  var sh = ss.getSheetByName(HOJA.CONCILIACION);
  if (sh) return sh;
  sh = ss.insertSheet(HOJA.CONCILIACION, 0);
  sh.getRange(1, 1, 1, COLS.length).setValues([COLS]).setFontWeight('bold').setBackground('#1f4e78')
    .setFontColor('#ffffff').setWrap(true).setVerticalAlignment('middle');
  sh.setFrozenRows(1);
  sh.setFrozenColumns(1);
  sh.setColumnWidth(C['Detalle Movimiento'] + 1, 260);
  sh.setColumnWidth(C['GLOSA'] + 1, 260);
  sh.setColumnWidth(C['NOMBRE CUENTA'] + 1, 200);
  sh.setColumnWidth(C['EXPLICACIÓN'] + 1, 420);
  sh.getRange(1, C['CUENTA CONTABLE'] + 1).setNote('Al escribir o elegir una cuenta, el nombre se completa solo y la fila pasa a MANUAL y APROBAR.');
  return sh;
}

/* ================================================================== */
/* Lectura de la historia y tablas auxiliares                         */
/* ================================================================== */

function normEnc_(s) { return mtNormTexto(s); }

/** Lee una hoja y devuelve {enc, idx(nombre)->col, filas}. */
function leerTabla_(nombre, filaEnc) {
  var sh = SpreadsheetApp.getActive().getSheetByName(nombre);
  if (!sh || sh.getLastRow() < 2) return null;
  var datos = sh.getDataRange().getValues();
  var fe = filaEnc || 0;
  var enc = datos[fe];
  var idx = {};
  enc.forEach(function (h, i) { var k = normEnc_(h); if (k && idx[k] === undefined) idx[k] = i; });
  return { hoja: sh, enc: enc, idx: idx, filas: datos.slice(fe + 1) };
}

function col_(t, nombre) {
  var i = t.idx[normEnc_(nombre)];
  return i === undefined ? -1 : i;
}

function val_(fila, i) { return i < 0 ? '' : fila[i]; }

function leerHistoria_() {
  var t = leerTabla_(HOJA.FLUJO);
  if (!t) throw new Error('La hoja FLUJO está vacía.');
  var c = {};
  ['EMPRESA', 'Fecha', 'Detalle Movimiento', 'Cheque o Cargo', 'Deposito o Abono', 'DETALLE 1', 'C.COSTO', 'GLOSA',
   'CUENTA CONTABLE', 'NOMBRE CUENTA', 'CENTRO COSTO', 'RUT', 'RAZON SOCIAL', 'categoria 4'].forEach(function (n) { c[n] = col_(t, n); });
  ['EMPRESA', 'Fecha', 'Detalle Movimiento', 'Cheque o Cargo', 'Deposito o Abono', 'CUENTA CONTABLE'].forEach(function (n) {
    if (c[n] < 0) throw new Error('A la hoja FLUJO le falta la columna "' + n + '".');
  });
  var hist = [], empresas = {}, nombres = {}, cat4 = {};
  t.filas.forEach(function (r) {
    var emp = r[c['EMPRESA']];
    if (!emp) return;
    empresas[String(emp).trim()] = 1;
    var cta = r[c['CUENTA CONTABLE']];
    if (cta !== '' && cta !== null) {
      var nc = val_(r, c['NOMBRE CUENTA']);
      if (nc && !nombres[cta]) nombres[cta] = nc;
      var c4 = val_(r, c['categoria 4']);
      if (c4 && String(c4).indexOf('#') !== 0) {
        var m = cat4[cta] = cat4[cta] || {};
        m[c4] = (m[c4] || 0) + 1;
      }
    }
    hist.push({
      empresa: emp, fecha: r[c['Fecha']], detalle: r[c['Detalle Movimiento']],
      cargo: r[c['Cheque o Cargo']], abono: r[c['Deposito o Abono']],
      detalle1: val_(r, c['DETALLE 1']), ccosto: val_(r, c['C.COSTO']), glosa: val_(r, c['GLOSA']),
      cuenta: cta, nombreCuenta: val_(r, c['NOMBRE CUENTA']), centro: val_(r, c['CENTRO COSTO']),
      rut: val_(r, c['RUT']), razon: val_(r, c['RAZON SOCIAL'])
    });
  });
  var cat4Top = {};
  for (var k in cat4) {
    var best = null, n = 0;
    for (var v in cat4[k]) if (cat4[k][v] > n) { n = cat4[k][v]; best = v; }
    cat4Top[k] = best;
  }
  return { hist: hist, empresas: Object.keys(empresas).sort(), nombresCuenta: nombres, cat4: cat4Top, tabla: t };
}

/** Plan de cuentas: hoja BBDD columnas CUENTA / NOMBRE (si existe) + nombres vistos en FLUJO. */
function leerPlanCuentas_(nombresFlujo) {
  var plan = {};
  for (var k in (nombresFlujo || {})) plan[k] = nombresFlujo[k];
  var t = leerTabla_(HOJA.PLAN);
  if (t) {
    var ic = col_(t, 'CUENTA'), inn = col_(t, 'NOMBRE');
    if (ic >= 0 && inn >= 0) t.filas.forEach(function (r) {
      if (r[ic] !== '' && r[inn]) plan[r[ic]] = r[inn];
    });
  }
  return plan;
}

function leerReglas_(plan) {
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.REGLAS);
  if (!sh || sh.getLastRow() < 2) return [];
  return sh.getRange(2, 1, sh.getLastRow() - 1, 14).getValues().filter(function (r) { return r[2] !== ''; }).map(function (r) {
    return {
      activa: r[0] === true || String(r[0]).toUpperCase() === 'SI', empresa: r[1], contiene: r[2], tipo: r[3],
      montoMin: r[4], montoMax: r[5],
      label: { detalle1: r[6], ccosto: r[7], glosa: r[8], cuenta: r[9], nombreCuenta: r[10] || (plan[r[9]] || ''),
               centro: r[11], rut: r[12], razon: r[13] }
    };
  });
}

function leerPagos_() {
  var t = leerTabla_(HOJA.PAGOS);
  if (!t) return [];
  var c = {
    fecha: col_(t, 'FECHA PAGO'), origen: col_(t, 'CUENTA ORIGEN'), rut: col_(t, 'RUT'), nombre: col_(t, 'NOMBRE'),
    monto: col_(t, 'MONTO'), asunto: col_(t, 'ASUNTO'), cuenta: col_(t, 'CÓDIGO CUENTA CONTABLE'),
    nombreCuenta: col_(t, 'NOMBRE CUENTA CONTABLE')
  };
  if (c.monto < 0 || c.nombre < 0 || c.cuenta < 0) return [];
  return t.filas.filter(function (r) { return r[c.monto] && r[c.cuenta] && String(r[c.cuenta]).indexOf('#') !== 0; }).map(function (r) {
    return {
      fecha: val_(r, c.fecha), fechaD: mtFechaPago(val_(r, c.fecha)), origen: val_(r, c.origen), rut: val_(r, c.rut),
      nombre: r[c.nombre], monto: r[c.monto], asunto: val_(r, c.asunto), cuenta: r[c.cuenta],
      nombreCuenta: val_(r, c.nombreCuenta), usado: false
    };
  });
}

function leerBaseFlujo_() {
  var t = leerTabla_(HOJA.BASE);
  var m = {};
  if (t) t.filas.forEach(function (r) { if (r[0] !== '' && !m[r[0]]) m[r[0]] = { cat2: r[1], cat3: r[3] }; });
  return m;
}

function normCuenta_(v) { return String(v === null || v === undefined ? '' : v).replace(/\D/g, '').replace(/^0+/, ''); }

/** N° de cuenta -> EMPRESA, desde la hoja CUENTAS BANCARIAS (confirmadas por el usuario). */
function leerCuentasBancarias_() {
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.CUENTAS);
  var m = {};
  if (sh && sh.getLastRow() > 1) sh.getRange(2, 1, sh.getLastRow() - 1, 2).getValues().forEach(function (r) {
    var k = normCuenta_(r[0]);
    if (k && r[1]) m[k] = String(r[1]);
  });
  return m;
}

/** N° de cuenta -> nombre corto, desde BBDD ("Cta. Provisión de Fondos" / "Cuenta"), p.ej. 886-05140-01 -> Octay. */
function leerCuentasBBDD_() {
  var m = {};
  var t = leerTabla_(HOJA.PLAN);
  if (!t) return m;
  var ic = -1, inn = -1, ie = col_(t, 'Empresa');
  t.enc.forEach(function (h, i) {
    var n = normEnc_(h);
    if (ic < 0 && /provision de fondos/.test(n)) ic = i;
  });
  if (ic < 0) return m;
  for (var i = ic + 1; i < t.enc.length; i++) if (normEnc_(t.enc[i]) === 'cuenta') { inn = i; break; }
  if (inn < 0) return m;
  t.filas.forEach(function (r) {
    var k = normCuenta_(r[ic]);
    if (k && r[inn]) m[k] = (ie >= 0 && r[ie] ? r[ie] + ' ' : '') + r[inn];
  });
  return m;
}

function guardarCuentaBancaria_(cuenta, empresa, titular, banco, rut) {
  if (!cuenta || !empresa) return;
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.CUENTAS);
  if (!sh) return;
  var n = sh.getLastRow();
  if (n > 1) {
    var datos = sh.getRange(2, 1, n - 1, 5).getValues();
    for (var i = 0; i < datos.length; i++) {
      if (normCuenta_(datos[i][0]) === cuenta) {
        sh.getRange(i + 2, 2).setValue(empresa);
        if (rut && !datos[i][4]) sh.getRange(i + 2, 5).setValue(rut);
        return;
      }
    }
  }
  sh.appendRow(["'" + cuenta, empresa, titular || '', banco || '', rut || '']);
}

var PALABRAS_GENERICAS_ = ['spa', 'ltda', 'sa', 'limitada', 'agricola', 'sociedad', 'inversiones', 'de', 'la', 'el', 'los', 'las', 'y'];

function tokensNombre_(s) {
  var t = mtTokens(s);
  var sig = t.filter(function (x) { return PALABRAS_GENERICAS_.indexOf(x) < 0; });
  // Sigla: "Tu Proyecto De Vida III" -> "tpdv" (+ "iii")
  var sigla = t.filter(function (x) { return ['spa', 'ltda', 'sa', 'limitada'].indexOf(x) < 0 && !/^[ivx]+$/.test(x); })
    .map(function (x) { return x[0]; }).join('');
  if (sigla.length >= 3) sig.push(sigla);
  return sig;
}

function puntajeNombre_(tk, empresa, banco) {
  var partes = String(empresa).split('-');
  var bancoE = mtNormTexto(partes.length > 1 ? partes[partes.length - 1] : '');
  var te = mtTokens(partes.length > 1 ? partes.slice(0, -1).join(' ') : empresa);
  var p = 0;
  tk.forEach(function (t) { if (te.indexOf(t) >= 0) p += t.length; });
  // Penaliza palabras de la empresa que no aparecen en el nombre (TPDV III vs TPDV)
  te.forEach(function (t) { if (tk.indexOf(t) < 0) p -= 0.5; });
  if (p > 0 && banco && bancoE.indexOf(mtNormTexto(banco)) >= 0) p += 2;
  return p;
}

/** Sugiere la empresa (etiqueta de FLUJO) de un bloque/cuenta de la cartola. */
function adivinarEmpresa_(info, empresas, cuentas, cuentasBBDD) {
  if (info.cuenta && cuentas[info.cuenta]) return { empresa: cuentas[info.cuenta], seguro: true };
  var nombres = [];
  if (info.cuenta && cuentasBBDD && cuentasBBDD[info.cuenta]) nombres.push(cuentasBBDD[info.cuenta]);
  nombres.push(info.titular);
  for (var i = 0; i < nombres.length; i++) {
    var tk = tokensNombre_(nombres[i]);
    var mejor = '', pts = 0;
    empresas.forEach(function (e) {
      var p = puntajeNombre_(tk, e, info.banco);
      if (p > pts) { pts = p; mejor = e; }
    });
    if (mejor) return { empresa: mejor, seguro: false };
  }
  return { empresa: '', seguro: false };
}

/* ================================================================== */
/* Funciones llamadas desde el diálogo                                 */
/* ================================================================== */

/** Analiza una cartola subida (matriz de celdas): cuentas que trae y empresa sugerida para cada una. */
function analizarCartola(nombre, filas) {
  var info = parsearCartola(filas, nombre);
  var empresas = listaEmpresas_();
  var cuentas = leerCuentasBancarias_(), cbbdd = leerCuentasBBDD_();
  return {
    nombre: nombre,
    advertencias: info.advertencias,
    empresas: empresas,
    bloques: info.bloques.map(function (b, i) {
      var sug = adivinarEmpresa_(b, empresas, cuentas, cbbdd);
      var fechas = b.movimientos.map(function (m) { return m.fecha.getTime(); });
      return {
        i: i, titular: b.titular, rut: b.rut, cuenta: b.cuenta, banco: b.banco, n: b.movimientos.length,
        desde: fechas.length ? mtClaveFecha(new Date(Math.min.apply(null, fechas))) : '',
        hasta: fechas.length ? mtClaveFecha(new Date(Math.max.apply(null, fechas))) : '',
        cargos: b.movimientos.reduce(function (s, m) { return s + m.cargo; }, 0),
        abonos: b.movimientos.reduce(function (s, m) { return s + m.abono; }, 0),
        advertencias: b.advertencias, empresaSugerida: sug.empresa, seguro: sug.seguro
      };
    })
  };
}

function listaEmpresas_() {
  var t = leerTabla_(HOJA.FLUJO);
  var ie = t ? col_(t, 'EMPRESA') : -1;
  if (ie < 0) return [];
  var e = {};
  t.filas.forEach(function (r) { if (r[ie]) e[String(r[ie]).trim()] = 1; });
  return Object.keys(e).sort();
}

/**
 * Procesa una o varias cartolas: [{nombre, filas, empresas: {indiceBloque: EMPRESA}}]
 * Los bloques (cuentas) sin movimientos o sin empresa asignada se omiten.
 */
function procesarCartolas(archivos) {
  var lotes = [];
  archivos.forEach(function (a) {
    var info = parsearCartola(a.filas, a.nombre);
    info.bloques.forEach(function (b, i) {
      var emp = (a.empresas || {})[i];
      if (!b.movimientos.length || !emp) return;
      guardarCuentaBancaria_(b.cuenta, emp, b.titular, b.banco, b.rut);
      lotes.push({ nombre: a.nombre + (b.cuenta ? ' · cta ' + b.cuenta : ''), empresa: emp, movimientos: b.movimientos });
    });
  });
  if (!lotes.length) throw new Error('No hay movimientos con empresa asignada para procesar.');
  return clasificarYEscribir_(lotes);
}

/** Menú 1b: procesa la cartola pegada en la hoja CARTOLA (una o varias cuentas). */
function procesarHojaCartola() {
  configurarSiFalta_();
  var ui = SpreadsheetApp.getUi();
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.CARTOLA);
  var info = parsearCartola(sh.getDataRange().getValues(), '');
  var bloques = info.bloques.filter(function (b) { return b.movimientos.length; });
  if (!bloques.length) { ui.alert('No se encontraron movimientos.\n' + info.advertencias.join('\n')); return; }
  var empresas = listaEmpresas_();
  var cuentas = leerCuentasBancarias_(), cbbdd = leerCuentasBBDD_();
  var lotes = [];
  for (var i = 0; i < bloques.length; i++) {
    var b = bloques[i];
    var sug = adivinarEmpresa_(b, empresas, cuentas, cbbdd);
    var empresa = sug.empresa;
    if (!sug.seguro) {
      var r = ui.prompt('Empresa de la cuenta ' + (i + 1) + ' de ' + bloques.length,
        'Titular: ' + b.titular + '   Cuenta: ' + b.cuenta + '   (' + b.movimientos.length + ' movimientos)\n\n' +
        'Escriba la EMPRESA tal como aparece en FLUJO.' + (empresa ? '\nDeje vacío para usar la sugerida: ' + empresa : '') +
        '\nEscriba OMITIR para no procesar esta cuenta.\n\nOpciones: ' + empresas.join(', '), ui.ButtonSet.OK_CANCEL);
      if (r.getSelectedButton() !== ui.Button.OK) return;
      var txt = r.getResponseText().trim();
      if (/^omitir$/i.test(txt)) continue;
      if (txt) {
        var e = empresas.filter(function (x) { return mtNormTexto(x) === mtNormTexto(txt); })[0];
        if (!e) { ui.alert('La empresa "' + txt + '" no existe en FLUJO. Se omite esta cuenta.'); continue; }
        empresa = e;
      }
      if (!empresa) continue;
      guardarCuentaBancaria_(b.cuenta, empresa, b.titular, b.banco, b.rut);
      cuentas[b.cuenta] = empresa;
    }
    if (sug.seguro) guardarCuentaBancaria_(b.cuenta, empresa, b.titular, b.banco, b.rut);
    lotes.push({ nombre: 'Hoja ' + HOJA.CARTOLA + (b.cuenta ? ' · cta ' + b.cuenta : ''), empresa: empresa, movimientos: b.movimientos });
  }
  if (!lotes.length) return;
  var res = clasificarYEscribir_(lotes);
  var rf = hayFacturas_() ? cruzarFacturas_() : null;
  ui.alert(textoResumen_(res) + (rf ? '\n\n' + textoResumenFacturas_(rf) : ''));
}

function textoResumen_(res) {
  return 'Movimientos procesados: ' + res.total + '\n\n' +
    '✅ Identificados: ' + res.porEstado[ESTADO.IDENTIFICADO] + '\n' +
    '🟡 Por revisar: ' + res.porEstado[ESTADO.REVISAR] + '\n' +
    '🔴 Pendientes (manual): ' + res.porEstado[ESTADO.PENDIENTE] + '\n' +
    '⚪ Ya cargados antes (FLUJO o CONCILIACION): ' + res.porEstado[ESTADO.DUPLICADO] + '\n\n' +
    'Revise la hoja ' + HOJA.CONCILIACION + '.';
}

/* ================================================================== */
/* Clasificación                                                       */
/* ================================================================== */

function clasificarYEscribir_(lotes) {
  var cfg = leerConfig_();
  var H = leerHistoria_();
  var plan = leerPlanCuentas_(H.nombresCuenta);
  var modelo = new ModeloConciliacion(H.hist, cfg);
  var reglas = leerReglas_(plan);
  var pagos = cfg.usarPagosPorHacer ? leerPagos_() : [];
  cfg.cuentasPropias = mtIndiceCuentasPropias(plan);
  cfg.cuentasBancarias = leerCuentasBancarias_();

  // Movimientos ya registrados (FLUJO y CONCILIACION) para no duplicar
  var existentes = {};
  H.hist.forEach(function (h) {
    var k = mtClaveDuplicado(h.empresa, h.fecha, h.detalle, h.cargo, h.abono);
    existentes[k] = (existentes[k] || 0) + 1;
  });
  var sh = hojaConciliacion_();
  if (sh.getLastRow() > 1) {
    sh.getRange(2, 1, sh.getLastRow() - 1, COLS.length).getValues().forEach(function (r) {
      var k = mtClaveDuplicado(r[C['EMPRESA']], r[C['Fecha']], r[C['Detalle Movimiento']], r[C['Cheque o Cargo']], r[C['Deposito o Abono']]);
      existentes['C§' + k] = (existentes['C§' + k] || 0) + 1;
    });
  }

  var salida = [], porEstado = {};
  [ESTADO.IDENTIFICADO, ESTADO.REVISAR, ESTADO.PENDIENTE, ESTADO.DUPLICADO].forEach(function (e) { porEstado[e] = 0; });
  var lote = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm');

  lotes.forEach(function (l) {
    l.movimientos.forEach(function (m) {
      m.empresa = l.empresa;
      var r = clasificarMovimiento_(m, modelo, reglas, pagos, plan, cfg, existentes);
      porEstado[r.estado]++;
      var lb = r.label || {};
      var fila = [];
      fila[C['ESTADO']] = r.estado;
      fila[C['CONFIANZA']] = r.confianza;
      fila[C['FUENTE']] = r.fuente;
      fila[C['EMPRESA']] = l.empresa;
      fila[C['Fecha']] = m.fecha;
      fila[C['Detalle Movimiento']] = m.detalle;
      fila[C['Cheque o Cargo']] = m.cargo;
      fila[C['Deposito o Abono']] = m.abono;
      fila[C['Saldo']] = m.saldo;
      fila[C['Docto. Nro.']] = m.docto;
      fila[C['Trn']] = m.trn;
      fila[C['Caja']] = m.caja;
      fila[C['Sucursal']] = m.sucursal;
      fila[C['DETALLE 1']] = lb.detalle1 || '';
      fila[C['C.COSTO']] = lb.ccosto || '';
      fila[C['GLOSA']] = lb.glosa || '';
      fila[C['CUENTA CONTABLE']] = lb.cuenta === undefined ? '' : lb.cuenta;
      fila[C['NOMBRE CUENTA']] = lb.nombreCuenta || (lb.cuenta ? plan[lb.cuenta] || '' : '');
      fila[C['CENTRO COSTO']] = lb.centro || '';
      fila[C['RUT']] = lb.rut || '';
      fila[C['RAZON SOCIAL']] = lb.razon || '';
      fila[C['APROBAR']] = r.estado === ESTADO.IDENTIFICADO;
      fila[C['EXPLICACIÓN']] = r.explicacion;
      fila[C['LOTE']] = lote + ' · ' + l.nombre;
      salida.push(fila);
    });
  });

  escribirConciliacion_(sh, salida, plan, H);
  SpreadsheetApp.getActive().setActiveSheet(sh);
  return { total: salida.length, porEstado: porEstado };
}

function clasificarMovimiento_(m, modelo, reglas, pagos, plan, cfg, existentes) {
  // 0. ¿Ya está en FLUJO / CONCILIACION?
  var k = mtClaveDuplicado(m.empresa, m.fecha, m.detalle, m.cargo, m.abono);
  if (existentes[k] > 0) {
    existentes[k]--;
    return { estado: ESTADO.DUPLICADO, confianza: 1, fuente: 'DUPLICADO', label: null,
             explicacion: 'Este movimiento (misma empresa, fecha, detalle y monto) ya está en FLUJO. No se traspasará.' };
  }
  if (existentes['C§' + k] > 0) {
    existentes['C§' + k]--;
    return { estado: ESTADO.DUPLICADO, confianza: 1, fuente: 'DUPLICADO', label: null,
             explicacion: 'Este movimiento ya se había cargado en CONCILIACION (fila anterior). No se traspasará esta copia.' };
  }

  // 1. Reglas manuales
  var rg = mtAplicarReglas(reglas, m);
  if (rg) {
    var lb = {};
    for (var x in rg.label) lb[x] = rg.label[x];
    if (!lb.ccosto && !lb.centro) { var hc = modelo.ccostoHabitual(m.empresa); lb.ccosto = hc.ccosto; lb.centro = hc.centro; }
    return { estado: ESTADO.IDENTIFICADO, confianza: 1, fuente: 'REGLA ' + (rg.regla + 2), label: lb,
             explicacion: 'Coincide con la regla de la fila ' + (rg.regla + 2) + ' de la hoja REGLAS.' };
  }

  // 2. Modelo predictivo sobre la historia
  var p = modelo.predecir(m);

  // 3. Cruce con PAGOS POR HACER (cargos)
  if (pagos.length) {
    var cp = mtCruzarPago(pagos, m, cfg.diasPagoVentana || MOTOR_DEFAULTS.diasPagoVentana);
    if (cp) {
      var pg = cp.pago;
      var conf = cp.puntos >= 2 ? 0.92 : 0.82;
      if (conf >= p.confianza || !p.label || String(p.label.cuenta) === String(pg.cuenta)) {
        pg.usado = true;
        var mismo = p.label && String(p.label.cuenta) === String(pg.cuenta);
        var hc2 = mismo ? p.label : modelo.ccostoHabitual(m.empresa);
        var lb2 = {
          detalle1: mismo ? p.label.detalle1 : '', ccosto: hc2.ccosto, centro: hc2.centro,
          glosa: pg.asunto, cuenta: pg.cuenta, nombreCuenta: pg.nombreCuenta || plan[pg.cuenta] || '',
          rut: pg.rut, razon: pg.nombre
        };
        if (mismo) conf = Math.max(conf, p.confianza, 0.9);
        return {
          estado: conf >= (cfg.umbralIdentificado || MOTOR_DEFAULTS.umbralIdentificado) ? ESTADO.IDENTIFICADO : ESTADO.REVISAR,
          confianza: conf, fuente: 'PAGOS POR HACER', label: lb2,
          explicacion: 'Cargo de $' + Math.round(pg.monto).toLocaleString('es-CL') + ' coincide con pago programado (' + pg.fecha + ') a ' +
                       pg.nombre + ': "' + pg.asunto + '".' + (mismo ? ' La historia sugiere la misma cuenta.' : '')
        };
      }
    }
  }

  // 4. Traspaso entre cuentas propias: el detalle trae el N° de una cuenta que está en el plan de cuentas
  var cpi = mtCuentaPropia(cfg.cuentasPropias, m.detalle);
  if (cpi) {
    var umbral = cfg.umbralIdentificado || MOTOR_DEFAULTS.umbralIdentificado;
    var dueña = (cfg.cuentasBancarias || {})[cpi.numero];
    var desc = 'cuenta propia ' + cpi.numero + (dueña ? ' (' + dueña + ')' : '') + ' = ' + cpi.cuenta + ' ' + cpi.nombre;
    if (p.label && String(p.label.cuenta) === String(cpi.cuenta)) {
      p.confianza = Math.max(p.confianza, 0.85);
      p.estado = p.confianza >= umbral ? ESTADO.IDENTIFICADO : p.estado;
      p.explicacion = 'Traspaso con ' + desc + '; la historia confirma. ' + p.explicacion;
    } else if (p.confianza < umbral) {
      var hc3 = modelo.ccostoHabitual(m.empresa, cpi.cuenta);
      var abono = mtNum(m.abono) > mtNum(m.cargo);
      return {
        estado: ESTADO.REVISAR, confianza: 0.7, fuente: 'CUENTA PROPIA',
        label: { detalle1: 'Traspaso de Fondos', ccosto: hc3.ccosto, centro: hc3.centro,
                 glosa: 'Traspaso ' + (abono ? 'desde ' : 'a ') + (dueña || 'cuenta ' + cpi.numero),
                 cuenta: cpi.cuenta, nombreCuenta: cpi.nombre, rut: '', razon: '' },
        explicacion: 'Traspaso con ' + desc + '. Verifique si corresponde a un reparto o venta (la historia sugería ' +
                     (p.label ? p.label.cuenta + ' ' + (p.label.nombreCuenta || '') + ' con ' + Math.round(p.confianza * 100) + '%' : 'nada') + ').'
      };
    }
  }
  return p;
}

function escribirConciliacion_(sh, filas, plan, H) {
  if (!filas.length) return;
  var ini = sh.getLastRow() + 1;
  var n = filas.length;
  sh.getRange(ini, 1, n, COLS.length).setValues(filas);
  formatearFilas_(sh, ini, n, plan, H);
}

function formatearFilas_(sh, ini, n, plan, H) {
  if (n <= 0) return;
  sh.getRange(ini, C['Fecha'] + 1, n, 1).setNumberFormat('dd-mm-yyyy');
  sh.getRange(ini, C['Cheque o Cargo'] + 1, n, 3).setNumberFormat('#,##0');
  sh.getRange(ini, C['CONFIANZA'] + 1, n, 1).setNumberFormat('0%');
  sh.getRange(ini, C['APROBAR'] + 1, n, 1).insertCheckboxes();
  sh.getRange(ini, C['EXPLICACIÓN'] + 1, n, 1).setWrap(false).setFontColor('#555555');

  // Lista desplegable de cuentas contables (permite escribir otra)
  var cuentas = Object.keys(plan).sort();
  if (cuentas.length && cuentas.length <= 500) {
    sh.getRange(ini, C['CUENTA CONTABLE'] + 1, n, 1).setDataValidation(
      SpreadsheetApp.newDataValidation().requireValueInList(cuentas, true).setAllowInvalid(true).build());
  }
  if (H) {
    var cc = {}, ce = {};
    H.hist.forEach(function (h) { if (h.ccosto) cc[h.ccosto] = 1; if (h.centro) ce[h.centro] = 1; });
    var lcc = Object.keys(cc).sort(), lce = Object.keys(ce).sort();
    if (lcc.length && lcc.length <= 500) sh.getRange(ini, C['C.COSTO'] + 1, n, 1).setDataValidation(
      SpreadsheetApp.newDataValidation().requireValueInList(lcc, true).setAllowInvalid(true).build());
    if (lce.length && lce.length <= 500) sh.getRange(ini, C['CENTRO COSTO'] + 1, n, 1).setDataValidation(
      SpreadsheetApp.newDataValidation().requireValueInList(lce, true).setAllowInvalid(true).build());
  }
  colorear_(sh, ini, n);
}

function colorear_(sh, ini, n) {
  var est = sh.getRange(ini, 1, n, 1).getValues();
  var COLORES = colores_();
  var bg = est.map(function (r) { var c = COLORES[r[0]] || '#ffffff'; return COLS.map(function () { return c; }); });
  sh.getRange(ini, 1, n, COLS.length).setBackgrounds(bg);
}

/* ================================================================== */
/* Edición manual                                                      */
/* ================================================================== */

/** Al elegir/escribir una cuenta en CONCILIACION se completa el nombre y queda MANUAL + APROBAR. */
function onEdit(e) {
  var sh = e.range.getSheet();
  if (sh.getName() === HOJA.FACTURAS) { onEditFacturas_(e); return; }
  if (sh.getName() !== HOJA.CONCILIACION || e.range.getRow() < 2) return;
  var col = e.range.getColumn();
  var editables = [C['DETALLE 1'], C['C.COSTO'], C['GLOSA'], C['CUENTA CONTABLE'], C['NOMBRE CUENTA'],
                   C['CENTRO COSTO'], C['RUT'], C['RAZON SOCIAL']].map(function (x) { return x + 1; });
  var ultima = e.range.getLastColumn();
  var toca = editables.some(function (c) { return c >= col && c <= ultima; });
  if (!toca) return;

  var filas = e.range.getNumRows(), ini = e.range.getRow();
  var datos = sh.getRange(ini, 1, filas, COLS.length).getValues();
  var tocaCuenta = col <= C['CUENTA CONTABLE'] + 1 && ultima >= C['CUENTA CONTABLE'] + 1;
  var plan = tocaCuenta ? leerPlanCuentasCache_() : null;
  datos.forEach(function (r) {
    if (r[C['ESTADO']] === ESTADO.DUPLICADO) return;
    if (tocaCuenta && r[C['CUENTA CONTABLE']] !== '') {
      var nom = plan[r[C['CUENTA CONTABLE']]];
      if (nom) r[C['NOMBRE CUENTA']] = nom;
    }
    if (r[C['ESTADO']] !== ESTADO.IDENTIFICADO || tocaCuenta) {
      r[C['ESTADO']] = ESTADO.MANUAL;
      r[C['FUENTE']] = 'ANALISTA';
      r[C['APROBAR']] = r[C['CUENTA CONTABLE']] !== '';
    }
  });
  sh.getRange(ini, 1, filas, COLS.length).setValues(datos);
  colorear_(sh, ini, filas);
}

function leerPlanCuentasCache_() {
  var cache = CacheService.getDocumentCache();
  var s = cache.get('planCuentas');
  if (s) return JSON.parse(s);
  var nombres = {};
  var t = leerTabla_(HOJA.FLUJO);
  var ic = col_(t, 'CUENTA CONTABLE'), inn = col_(t, 'NOMBRE CUENTA');
  t.filas.forEach(function (r) { if (r[ic] !== '' && r[inn] && !nombres[r[ic]]) nombres[r[ic]] = r[inn]; });
  var plan = leerPlanCuentas_(nombres);
  try { cache.put('planCuentas', JSON.stringify(plan), 21600); } catch (err) { /* demasiado grande */ }
  return plan;
}

function filtrarPendientes() {
  var sh = hojaConciliacion_();
  if (sh.getLastRow() < 2) return;
  var f = sh.getFilter();
  if (f) f.remove();
  var rango = sh.getRange(1, 1, sh.getLastRow(), COLS.length);
  rango.createFilter().setColumnFilterCriteria(C['ESTADO'] + 1,
    SpreadsheetApp.newFilterCriteria().setHiddenValues([ESTADO.IDENTIFICADO, ESTADO.DUPLICADO, ESTADO.MANUAL]).build());
  SpreadsheetApp.getActive().setActiveSheet(sh);
}

function quitarFiltro() {
  var sh = hojaConciliacion_();
  var f = sh.getFilter();
  if (f) f.remove();
}

/** Crea una regla en REGLAS a partir de la fila seleccionada en CONCILIACION. */
function crearReglaDesdeSeleccion() {
  var ui = SpreadsheetApp.getUi();
  var sh = SpreadsheetApp.getActiveSheet();
  if (sh.getName() !== HOJA.CONCILIACION) { ui.alert('Seleccione una fila de la hoja ' + HOJA.CONCILIACION + '.'); return; }
  var fila = sh.getActiveRange().getRow();
  if (fila < 2) return;
  var r = sh.getRange(fila, 1, 1, COLS.length).getValues()[0];
  if (r[C['CUENTA CONTABLE']] === '') { ui.alert('La fila no tiene CUENTA CONTABLE.'); return; }
  var sugerido = String(r[C['Detalle Movimiento']]).replace(/\d{4,}/g, ' ').replace(/\s+/g, ' ').trim();
  var resp = ui.prompt('Nueva regla', 'Texto que debe contener el detalle (puede usar /regex/):', ui.ButtonSet.OK_CANCEL);
  if (resp.getSelectedButton() !== ui.Button.OK) return;
  var contiene = resp.getResponseText().trim() || sugerido;
  var reglas = SpreadsheetApp.getActive().getSheetByName(HOJA.REGLAS);
  reglas.appendRow([true, r[C['EMPRESA']], contiene, r[C['Deposito o Abono']] > r[C['Cheque o Cargo']] ? 'ABONO' : 'CARGO', '', '',
    r[C['DETALLE 1']], r[C['C.COSTO']], r[C['GLOSA']], r[C['CUENTA CONTABLE']], r[C['NOMBRE CUENTA']], r[C['CENTRO COSTO']],
    r[C['RUT']], r[C['RAZON SOCIAL']]]);
  reglas.getRange(reglas.getLastRow(), 1).insertCheckboxes().check();
  ui.alert('Regla creada en la hoja REGLAS (fila ' + reglas.getLastRow() + ') para "' + contiene + '" en ' + r[C['EMPRESA']] + '.');
}

/* ================================================================== */
/* Traspaso a FLUJO                                                    */
/* ================================================================== */

function traspasarAFlujo() {
  var ui = SpreadsheetApp.getUi();
  var sh = hojaConciliacion_();
  var n = sh.getLastRow() - 1;
  if (n < 1) { ui.alert('No hay movimientos en ' + HOJA.CONCILIACION + '.'); return; }
  quitarFiltro();
  // Antes de traspasar, aplicar los cruces con facturas confirmados (glosa, RUT, razón social)
  if (hayFacturas_()) cruzarFacturas_();
  var datos = sh.getRange(2, 1, n, COLS.length).getValues();

  var aprobados = [], quedan = [], sinCuenta = 0;
  datos.forEach(function (r) {
    if (r[C['ESTADO']] === ESTADO.DUPLICADO) { quedan.push(r); return; }
    if (r[C['APROBAR']] === true) {
      if (r[C['CUENTA CONTABLE']] === '' || r[C['CUENTA CONTABLE']] === null) { sinCuenta++; quedan.push(r); return; }
      aprobados.push(r);
    } else quedan.push(r);
  });
  if (!aprobados.length) {
    ui.alert('No hay filas con APROBAR marcado y CUENTA CONTABLE completa.' + (sinCuenta ? '\n' + sinCuenta + ' filas aprobadas no tienen cuenta.' : ''));
    return;
  }
  var pend = quedan.filter(function (r) { return r[C['ESTADO']] !== ESTADO.DUPLICADO; }).length;
  var ok = ui.alert('Traspasar a FLUJO',
    'Se agregarán ' + aprobados.length + ' movimientos a FLUJO.\n' +
    (pend ? pend + ' movimientos quedan en CONCILIACION (no aprobados o sin cuenta).\n' : '') +
    '¿Continuar?', ui.ButtonSet.YES_NO);
  if (ok !== ui.Button.YES) return;

  var H = leerHistoria_();
  var base = leerBaseFlujo_();
  var t = H.tabla;
  var enc = t.enc.map(normEnc_);
  var nuevas = aprobados.map(function (r) { return filaFlujo_(r, enc, H.cat4, base); });
  var flujo = t.hoja;
  var ini = flujo.getLastRow() + 1;
  flujo.getRange(ini, 1, nuevas.length, enc.length).setValues(nuevas);
  var iF = enc.indexOf(normEnc_('Fecha'));
  if (iF >= 0) flujo.getRange(ini, iF + 1, nuevas.length, 1).setNumberFormat('dd-mm-yyyy');

  // Reescribir CONCILIACION solo con lo que queda (sin los duplicados ya informados)
  var resto = quedan.filter(function (r) { return r[C['ESTADO']] !== ESTADO.DUPLICADO; });
  sh.getRange(2, 1, n, COLS.length).clearContent().clearDataValidations().setBackground(null);
  if (resto.length) {
    sh.getRange(2, 1, resto.length, COLS.length).setValues(resto);
    formatearFilas_(sh, 2, resto.length, leerPlanCuentas_(H.nombresCuenta), H);
  }
  CacheService.getDocumentCache().remove('planCuentas');
  ui.alert('Listo: ' + aprobados.length + ' movimientos agregados a FLUJO (desde la fila ' + ini + ').\n' +
    (resto.length ? resto.length + ' movimientos siguen pendientes en CONCILIACION.' : 'CONCILIACION quedó vacía.'));
}

/** Construye una fila de FLUJO respetando el orden de sus encabezados. */
function filaFlujo_(r, enc, cat4, base) {
  var fecha = mtFecha(r[C['Fecha']]);
  var cargo = mtNum(r[C['Cheque o Cargo']]), abono = mtNum(r[C['Deposito o Abono']]);
  var flujo = abono - cargo;
  var cta = r[C['CUENTA CONTABLE']];
  var c4 = cat4[cta] || '';
  var b = base[c4] || {};
  var v = {};
  v['empresa'] = r[C['EMPRESA']];
  v['fecha'] = fecha;
  v['detalle movimiento'] = r[C['Detalle Movimiento']];
  v['cheque o cargo'] = cargo;
  v['deposito o abono'] = abono;
  v['saldo'] = r[C['Saldo']];
  v['docto nro'] = r[C['Docto. Nro.']];
  v['trn'] = r[C['Trn']];
  v['caja'] = r[C['Caja']];
  v['sucursal'] = r[C['Sucursal']];
  v['detalle 1'] = r[C['DETALLE 1']];
  v['c costo'] = r[C['C.COSTO']];
  v['glosa'] = r[C['GLOSA']];
  v['cuenta contable'] = cta;
  v['nombre cuenta'] = r[C['NOMBRE CUENTA']];
  v['mes'] = fecha ? fecha.getFullYear() * 100 + fecha.getMonth() + 1 : '';
  v['concepto'] = r[C['GLOSA']];
  v['debe'] = cargo;
  v['haber'] = abono;
  v['centro costo'] = r[C['CENTRO COSTO']];
  v['rut'] = r[C['RUT']];
  v['razon social'] = r[C['RAZON SOCIAL']];
  v['categoria 1'] = flujo > 1 ? 'Ingreso' : 'Egreso';
  v['flujo'] = flujo;
  v['categoria 4'] = c4;
  v['mes ano'] = fecha ? (fecha.getMonth() + 1) + ' ' + fecha.getFullYear() : '';
  v['categoria 2'] = b.cat2 || '';
  v['categoria 3'] = b.cat3 || '';
  return enc.map(function (h) { return v[h] === undefined || v[h] === null ? '' : v[h]; });
}

function limpiarConciliacion() {
  var ui = SpreadsheetApp.getUi();
  if (ui.alert('Limpiar', '¿Borrar todos los movimientos de ' + HOJA.CONCILIACION + ' (no afecta FLUJO)?', ui.ButtonSet.YES_NO) !== ui.Button.YES) return;
  var sh = hojaConciliacion_();
  quitarFiltro();
  if (sh.getLastRow() > 1) sh.getRange(2, 1, sh.getLastRow() - 1, COLS.length).clearContent().clearDataValidations().setBackground(null);
}

/* ================================================================== */
/* Evaluación del modelo                                               */
/* ================================================================== */

/** Entrena con la historia anterior a cada uno de los últimos 3 meses y mide aciertos. */
function evaluarModelo() {
  var cfg = leerConfig_();
  var H = leerHistoria_();
  var hist = H.hist.filter(function (h) { return mtFecha(h.fecha) && h.cuenta !== '' && h.cuenta !== null; });
  var meses = {};
  hist.forEach(function (h) { meses[mtClaveFecha(h.fecha).slice(0, 7)] = 1; });
  var lista = Object.keys(meses).sort().slice(-3);
  var tot = {};
  [ESTADO.IDENTIFICADO, ESTADO.REVISAR, ESTADO.PENDIENTE].forEach(function (e) { tot[e] = { n: 0, ok: 0 }; });
  var N = 0;
  lista.forEach(function (mes) {
    var train = hist.filter(function (h) { return mtClaveFecha(h.fecha).slice(0, 7) < mes; });
    var test = hist.filter(function (h) { return mtClaveFecha(h.fecha).slice(0, 7) === mes; });
    var m = new ModeloConciliacion(train, cfg);
    test.forEach(function (h) {
      if (/saldo caja inicial/i.test(String(h.detalle))) return;
      var p = m.predecir(h);
      N++;
      tot[p.estado].n++;
      if (p.label && String(p.label.cuenta) === String(h.cuenta)) tot[p.estado].ok++;
    });
  });
  var txt = 'Prueba sobre ' + N + ' movimientos de ' + lista.join(', ') +
    ' (entrenando solo con los meses anteriores a cada uno):\n\n';
  [ESTADO.IDENTIFICADO, ESTADO.REVISAR, ESTADO.PENDIENTE].forEach(function (e) {
    var x = tot[e];
    txt += e + ': ' + pct_(x.n, N) + ' de los movimientos · cuenta correcta en ' + pct_(x.ok, x.n) + '\n';
  });
  txt += '\nSi IDENTIFICADO tiene demasiados errores, suba "umbralIdentificado" en CONFIG; si quedan demasiados PENDIENTES, bájelo con cuidado o agregue REGLAS.';
  SpreadsheetApp.getUi().alert('Precisión del modelo', txt, SpreadsheetApp.getUi().ButtonSet.OK);
}

function pct_(a, b) { return (b ? Math.round(a / b * 1000) / 10 : 0) + '%'; }
