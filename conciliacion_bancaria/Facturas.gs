/**
 * Facturas.gs — Carga del Registro de Compras del SII y cruce con los pagos.
 *
 * Hoja FACTURAS COMPRA: una fila por documento del SII (factura, nota de
 * crédito, etc.) con el movimiento bancario que lo paga:
 *   CONCILIADA     pago encontrado con alta confianza (se aplica solo a CONCILIACION)
 *   CONFIRMADA     el analista marcó CONFIRMAR
 *   YA REGISTRADA  el pago ya está en FLUJO y su glosa menciona el folio
 *   POR REVISAR    hay una propuesta, falta confirmarla (casilla CONFIRMAR)
 *   SIN PAGO       no se encontró el pago (factura por pagar)
 *   NC SIN APLICAR nota de crédito aún no descontada en un pago
 *
 * Al aplicar un cruce, la fila de CONCILIACION queda con GLOSA "Pago FC.<folio>
 * <razón social>", RUT y RAZÓN SOCIAL del proveedor y la cuenta de proveedores.
 */

var FCOLS = ['ESTADO', 'CONFIANZA', 'CONFIRMAR', 'EMPRESA RUT', 'EMPRESA', 'PERIODO', 'TIPO DOC', 'FOLIO', 'RUT PROVEEDOR',
  'RAZON SOCIAL', 'FECHA DOCTO', 'MONTO NETO', 'IVA', 'MONTO TOTAL', 'MONTO A PAGAR', 'FECHA PAGO', 'CUENTA DE PAGO',
  'DETALLE TRANSFERENCIA', 'MONTO TRANSFERIDO', 'UBICACIÓN', 'PAGO CONJUNTO CON', 'EXPLICACIÓN', 'CLAVE MOV',
  'MOVS RECHAZADOS', 'CLAVE DOC', 'ARCHIVO'];
var FC = {};
FCOLS.forEach(function (n, i) { FC[n] = i; });

var F_FIJOS = ['CONFIRMADA', 'CONCILIADA', 'YA REGISTRADA'];
var F_SEP = ' + ';

function coloresFacturas_() {
  return { 'CONCILIADA': '#d9ead3', 'CONFIRMADA': '#cfe2f3', 'YA REGISTRADA': '#d9d9d9', 'POR REVISAR': '#fff2cc',
           'SIN PAGO': '#f4cccc', 'NC SIN APLICAR': '#fce5cd' };
}

function hayFacturas_() {
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.FACTURAS);
  return !!(sh && sh.getLastRow() > 1);
}

function hojaFacturas_() {
  var ss = SpreadsheetApp.getActive();
  var sh = ss.getSheetByName(HOJA.FACTURAS);
  if (sh) return sh;
  sh = ss.insertSheet(HOJA.FACTURAS);
  sh.getRange(1, 1, 1, FCOLS.length).setValues([FCOLS]).setFontWeight('bold').setBackground('#38761d')
    .setFontColor('#ffffff').setWrap(true).setVerticalAlignment('middle');
  sh.setFrozenRows(1);
  sh.setFrozenColumns(1);
  sh.setColumnWidth(FC['RAZON SOCIAL'] + 1, 240);
  sh.setColumnWidth(FC['DETALLE TRANSFERENCIA'] + 1, 260);
  sh.setColumnWidth(FC['EXPLICACIÓN'] + 1, 420);
  sh.getRange(1, FC['CONFIRMAR'] + 1).setNote('Marque para confirmar el pago propuesto (se aplica a CONCILIACION al cruzar o traspasar).\n' +
    'Desmarque una CONCILIADA para rechazar ese movimiento: el próximo cruce buscará otro.');
  return sh;
}

/** RUT de la empresa -> etiquetas de EMPRESA de FLUJO (desde CUENTAS BANCARIAS). */
function leerRutEmpresas_() {
  var m = {};
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.CUENTAS);
  if (!sh || sh.getLastRow() < 2 || sh.getLastColumn() < 5) return m;
  sh.getRange(2, 1, sh.getLastRow() - 1, 5).getValues().forEach(function (r) {
    var rut = normRut_(r[4]);
    if (!rut || !r[1]) return;
    var l = m[rut] = m[rut] || [];
    if (l.indexOf(String(r[1])) < 0) l.push(String(r[1]));
  });
  return m;
}

function normRut_(r) { return String(r || '').replace(/[^\dkK\-]/g, '').toUpperCase(); }

/** Asocia un RUT a una empresa cuando no hay cartola que lo indique (fila sin N° de cuenta). */
function guardarRutEmpresa_(rut, empresa) {
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.CUENTAS);
  if (!sh || !rut || !empresa) return;
  sh.appendRow(['', empresa, '(asignada al cargar el RCV del SII)', '', rut]);
}

/* ================================================================== */
/* Carga del RCV                                                       */
/* ================================================================== */

/** Llamada desde el diálogo: resume el archivo del SII y en qué empresas se buscarán los pagos. */
function analizarRcv(nombre, filas) {
  var info = parsearRcv(filas, nombre);
  var rutEmp = leerRutEmpresas_();
  var existentes = clavesDocsExistentes_();
  var nuevos = info.documentos.filter(function (d) { return !existentes[siiClaveDoc(info.rutEmpresa, d)]; }).length;
  var fechas = info.documentos.map(function (d) { return d.fechaDocto ? d.fechaDocto.getTime() : null; }).filter(function (x) { return x; });
  return {
    nombre: nombre, rutEmpresa: info.rutEmpresa, periodo: info.periodo, libro: info.libro, estadoRcv: info.estadoRcv,
    n: info.documentos.length, nuevos: nuevos,
    total: info.documentos.reduce(function (s, d) { return s + d.montoPagar; }, 0),
    desde: fechas.length ? mtClaveFecha(new Date(Math.min.apply(null, fechas))) : '',
    hasta: fechas.length ? mtClaveFecha(new Date(Math.max.apply(null, fechas))) : '',
    proveedores: Object.keys(info.documentos.reduce(function (o, d) { o[d.rutProveedor] = 1; return o; }, {})).length,
    empresasAsignadas: rutEmp[normRut_(info.rutEmpresa)] || [],
    empresas: listaEmpresas_(), advertencias: info.advertencias
  };
}

function clavesDocsExistentes_() {
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.FACTURAS);
  var m = {};
  if (sh && sh.getLastRow() > 1) sh.getRange(2, FC['CLAVE DOC'] + 1, sh.getLastRow() - 1, 1).getValues()
    .forEach(function (r) { if (r[0]) m[r[0]] = 1; });
  return m;
}

/**
 * Agrega los documentos de uno o más RCV a FACTURAS COMPRA (sin duplicar).
 * @param {Array} lista [{nombre, filas, empresa}]  empresa: elegida en el diálogo si el RUT no tenía cuentas asociadas
 */
function procesarRcv_(lista) {
  var sh = hojaFacturas_();
  var existentes = clavesDocsExistentes_();
  var rutEmp = leerRutEmpresas_();
  var nuevas = [], repetidos = 0, advert = [];
  lista.forEach(function (a) {
    var info = parsearRcv(a.filas, a.nombre);
    advert = advert.concat(info.advertencias.map(function (x) { return a.nombre + ': ' + x; }));
    if (info.libro !== 'COMPRA') return;
    var rut = normRut_(info.rutEmpresa);
    var emps = rutEmp[rut] || [];
    if (a.empresa && emps.indexOf(a.empresa) < 0) {
      if (rut) guardarRutEmpresa_(rut, a.empresa);
      emps = emps.concat([a.empresa]);
      rutEmp[rut] = emps;
    }
    info.documentos.forEach(function (d) {
      var clave = siiClaveDoc(rut, d);
      if (existentes[clave]) { repetidos++; return; }
      existentes[clave] = 1;
      var f = FCOLS.map(function () { return ''; });
      f[FC['ESTADO']] = ''; f[FC['CONFIRMAR']] = false;
      f[FC['EMPRESA RUT']] = rut; f[FC['EMPRESA']] = emps.join(', ');
      f[FC['PERIODO']] = info.periodo; f[FC['TIPO DOC']] = d.tipoDoc + ' ' + d.tipoNombre;
      f[FC['FOLIO']] = d.folio; f[FC['RUT PROVEEDOR']] = d.rutProveedor; f[FC['RAZON SOCIAL']] = d.razonSocial;
      f[FC['FECHA DOCTO']] = d.fechaDocto || ''; f[FC['MONTO NETO']] = d.neto; f[FC['IVA']] = d.iva;
      f[FC['MONTO TOTAL']] = d.total; f[FC['MONTO A PAGAR']] = d.montoPagar;
      f[FC['CLAVE DOC']] = clave; f[FC['ARCHIVO']] = a.nombre;
      nuevas.push(f);
    });
  });
  if (nuevas.length) {
    var ini = sh.getLastRow() + 1;
    sh.getRange(ini, 1, nuevas.length, FCOLS.length).setValues(nuevas);
    sh.getRange(ini, FC['CONFIRMAR'] + 1, nuevas.length, 1).insertCheckboxes();
    sh.getRange(ini, FC['FECHA DOCTO'] + 1, nuevas.length, 1).setNumberFormat('dd-mm-yyyy');
    sh.getRange(ini, FC['FECHA PAGO'] + 1, nuevas.length, 1).setNumberFormat('dd-mm-yyyy');
    sh.getRange(ini, FC['MONTO NETO'] + 1, nuevas.length, 4).setNumberFormat('#,##0');
    sh.getRange(ini, FC['MONTO TRANSFERIDO'] + 1, nuevas.length, 1).setNumberFormat('#,##0');
    sh.getRange(ini, FC['CONFIANZA'] + 1, nuevas.length, 1).setNumberFormat('0%');
  }
  return { nuevos: nuevas.length, repetidos: repetidos, advertencias: advert };
}

/**
 * Llamada desde el diálogo con cartolas y/o archivos RCV.
 * payload: {cartolas: [{nombre, filas, empresas}], rcv: [{nombre, filas, empresa}]}
 */
function procesarArchivos(payload) {
  configurarSiFalta_();
  var out = { cartolas: null, facturas: null, carga: null };
  if (payload.cartolas && payload.cartolas.length) out.cartolas = procesarCartolas(payload.cartolas);
  if (payload.rcv && payload.rcv.length) out.carga = procesarRcv_(payload.rcv);
  if (hayFacturas_()) out.facturas = cruzarFacturas_();
  return out;
}

/* ================================================================== */
/* Cruce                                                               */
/* ================================================================== */

/** Menú 4. */
function cruzarFacturas() {
  var ui = SpreadsheetApp.getUi();
  if (!hayFacturas_()) { ui.alert('Primero cargue el Registro de Compras del SII desde el menú 1.'); return; }
  var r = cruzarFacturas_();
  SpreadsheetApp.getActive().setActiveSheet(hojaFacturas_());
  ui.alert(textoResumenFacturas_(r));
}

function textoResumenFacturas_(r) {
  var e = r.porEstado;
  return 'Facturas SII (' + r.total + ' documentos):\n' +
    '✅ Pagadas (conciliadas/confirmadas): ' + ((e['CONCILIADA'] || 0) + (e['CONFIRMADA'] || 0)) + '\n' +
    '⚪ Ya registradas en FLUJO: ' + (e['YA REGISTRADA'] || 0) + '\n' +
    '🟡 Pago propuesto por revisar: ' + (e['POR REVISAR'] || 0) + '\n' +
    '🔴 Sin pago encontrado: ' + (e['SIN PAGO'] || 0) + (e['NC SIN APLICAR'] ? '\n🟠 Notas de crédito sin aplicar: ' + e['NC SIN APLICAR'] : '') + '\n' +
    'Movimientos de CONCILIACION completados con datos de la factura: ' + r.aplicados + '.';
}

/**
 * Movimientos (cargos) de FLUJO y CONCILIACION con una clave estable:
 * misma empresa+fecha+detalle+monto (+ n° de repetición).
 */
function leerMovimientosCruce_() {
  var lista = [], porClave = {}, cuenta = {};
  function agregar(m) {
    var base = mtClaveDuplicado(m.empresa, m.fecha, m.detalle, m.cargo, m.abono);
    cuenta[base] = (cuenta[base] || 0) + 1;
    m.clave = base + '#' + cuenta[base];
    porClave[m.clave] = m;
    if (mtNum(m.cargo) > 0) lista.push(m);
  }
  var t = leerTabla_(HOJA.FLUJO);
  var c = {};
  ['EMPRESA', 'Fecha', 'Detalle Movimiento', 'Cheque o Cargo', 'Deposito o Abono', 'GLOSA', 'RUT', 'RAZON SOCIAL', 'CUENTA CONTABLE']
    .forEach(function (n) { c[n] = col_(t, n); });
  var hist = [];
  t.filas.forEach(function (r, i) {
    if (!r[c['EMPRESA']] || !r[c['Fecha']]) return;
    var m = { origen: 'FLUJO', fila: i + 2, empresa: r[c['EMPRESA']], fecha: r[c['Fecha']], detalle: r[c['Detalle Movimiento']],
      cargo: mtNum(r[c['Cheque o Cargo']]), abono: mtNum(r[c['Deposito o Abono']]), glosa: val_(r, c['GLOSA']), rut: val_(r, c['RUT']),
      razon: val_(r, c['RAZON SOCIAL']) };
    agregar(m);
    if (m.cargo) hist.push({ detalle: m.detalle, cargo: m.cargo, glosa: m.glosa, rut: m.rut, razon: m.razon });
  });
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.CONCILIACION);
  if (sh && sh.getLastRow() > 1) {
    sh.getRange(2, 1, sh.getLastRow() - 1, COLS.length).getValues().forEach(function (r, i) {
      if (r[C['ESTADO']] === ESTADO.DUPLICADO || !r[C['EMPRESA']]) return;
      agregar({ origen: 'CONCILIACION', fila: i + 2, empresa: r[C['EMPRESA']], fecha: r[C['Fecha']], detalle: r[C['Detalle Movimiento']],
        cargo: mtNum(r[C['Cheque o Cargo']]), abono: mtNum(r[C['Deposito o Abono']]), glosa: r[C['GLOSA']], rut: r[C['RUT']],
        razon: r[C['RAZON SOCIAL']], estado: r[C['ESTADO']] });
    });
  }
  return { lista: lista, porClave: porClave, historia: hist };
}

/** Recalcula los cruces de las facturas no confirmadas y aplica los confirmados a CONCILIACION. */
function cruzarFacturas_() {
  var sh = hojaFacturas_();
  var n = sh.getLastRow() - 1;
  var res = { total: 0, porEstado: {}, aplicados: 0 };
  if (n < 1) return res;
  var filas = sh.getRange(2, 1, n, FCOLS.length).getValues();
  var cfg = leerConfig_();
  var M = leerMovimientosCruce_();
  var rutEmp = leerRutEmpresas_();

  // 1. Las fijas (confirmadas/conciliadas) reservan sus movimientos
  var usadas = {}, pendientes = [];
  filas.forEach(function (r, i) {
    var claves = String(r[FC['CLAVE MOV']] || '').split(F_SEP).filter(String);
    var fija = F_FIJOS.indexOf(r[FC['ESTADO']]) >= 0 && claves.length && claves.every(function (k) { return M.porClave[k]; });
    if (fija) { claves.forEach(function (k) { usadas[k] = 1; }); return; }
    var tipo = parseInt(r[FC['TIPO DOC']], 10);
    var emps = (rutEmp[normRut_(r[FC['EMPRESA RUT']])] || []).slice();
    String(r[FC['EMPRESA']] || '').split(',').map(function (x) { return x.trim(); }).filter(String)
      .forEach(function (x) { if (emps.indexOf(x) < 0) emps.push(x); });
    pendientes.push({
      id: i, fila: i, tipoDoc: tipo, abrev: SII_ABREV[tipo] || 'DOC', folio: String(r[FC['FOLIO']]),
      rutProveedor: String(r[FC['RUT PROVEEDOR']]), razonSocial: String(r[FC['RAZON SOCIAL']]),
      fechaDocto: mtFecha(r[FC['FECHA DOCTO']]), montoPagar: mtNum(r[FC['MONTO A PAGAR']]), empresas: emps,
      rechazados: String(r[FC['MOVS RECHAZADOS']] || '').split(F_SEP).filter(String)
    });
  });

  // 2. Cruce de las pendientes (cada una sin sus movimientos rechazados)
  if (pendientes.length) {
    var op = {};
    ['umbralFactura', 'umbralFacturaRevisar', 'diasAntes', 'diasDespues'].forEach(function (k) { if (cfg[k] !== undefined) op[k] = cfg[k]; });
    if (cfg.buscarOtrasEmpresas !== undefined) op.buscarOtrasEmpresas = cfg.buscarOtrasEmpresas;
    var cruce = new CruceFacturas(M.lista, M.historia, op);
    var conRech = pendientes.filter(function (d) { return d.rechazados.length; });
    var r1 = cruce.cruzar(pendientes.filter(function (d) { return !d.rechazados.length; }), usadas);
    conRech.forEach(function (d) {
      var u = {}; for (var k in usadas) u[k] = 1;
      for (var id in r1) r1[id].movs.forEach(function (m) { u[m.clave] = 1; });
      d.rechazados.forEach(function (k) { u[k] = 1; });
      var r2 = cruce.cruzar([d], u);
      r1[d.id] = r2[d.id];
    });
    pendientes.forEach(function (d) {
      var x = r1[d.id], r = filas[d.fila];
      r[FC['ESTADO']] = x.estado;
      r[FC['CONFIANZA']] = x.confianza;
      r[FC['CONFIRMAR']] = x.estado === 'CONCILIADA' || x.estado === 'YA REGISTRADA';
      r[FC['CLAVE MOV']] = x.movs.map(function (m) { return m.clave; }).join(F_SEP);
      r[FC['PAGO CONJUNTO CON']] = x.docsGrupo.map(function (id) {
        var o = filas[id];
        return (SII_ABREV[parseInt(o[FC['TIPO DOC']], 10)] || 'DOC') + '.' + o[FC['FOLIO']] + ' $' + siiFmt_(mtNum(o[FC['MONTO A PAGAR']]));
      }).join(', ');
      r[FC['EXPLICACIÓN']] = x.explicacion;
    });
  }

  // 3. Datos del movimiento (también para las fijas: su ubicación cambia al traspasar a FLUJO)
  filas.forEach(function (r) {
    var movs = String(r[FC['CLAVE MOV']] || '').split(F_SEP).filter(String).map(function (k) { return M.porClave[k]; }).filter(Boolean);
    r[FC['FECHA PAGO']] = movs.length ? mtFecha(movs[0].fecha) : '';
    r[FC['CUENTA DE PAGO']] = movs.map(function (m) { return m.empresa; }).filter(function (v, i, a) { return a.indexOf(v) === i; }).join(', ');
    r[FC['DETALLE TRANSFERENCIA']] = movs.map(function (m) { return String(m.detalle).trim(); }).join(F_SEP);
    r[FC['MONTO TRANSFERIDO']] = movs.length ? movs.reduce(function (s, m) { return s + mtNum(m.cargo); }, 0) : '';
    r[FC['UBICACIÓN']] = movs.map(function (m) { return m.origen + ' fila ' + m.fila; }).join(F_SEP);
    res.porEstado[r[FC['ESTADO']]] = (res.porEstado[r[FC['ESTADO']]] || 0) + 1;
  });
  res.total = filas.length;
  sh.getRange(2, 1, n, FCOLS.length).setValues(filas);
  var col = coloresFacturas_();
  sh.getRange(2, 1, n, FCOLS.length).setBackgrounds(filas.map(function (r) { var c = col[r[FC['ESTADO']]] || '#ffffff'; return FCOLS.map(function () { return c; }); }));
  sh.getRange(2, FC['FECHA PAGO'] + 1, n, 1).setNumberFormat('dd-mm-yyyy');

  // 4. Aplicar a CONCILIACION / FLUJO
  res.aplicados = aplicarCruces_(filas, M, cfg);
  return res;
}

/** Completa CONCILIACION (y RUT/razón social vacíos en FLUJO) con las facturas pagadas. */
function aplicarCruces_(filas, M, cfg) {
  var porMov = {};      // clave mov -> {docs:[fila], estado, cuotas}
  filas.forEach(function (r) {
    var claves = String(r[FC['CLAVE MOV']] || '').split(F_SEP).filter(String);
    if (!claves.length) return;
    claves.forEach(function (k) {
      var x = porMov[k] = porMov[k] || { docs: [], estados: [], cuotas: claves.length > 1 };
      x.docs.push(r); x.estados.push(r[FC['ESTADO']]);
    });
  });
  var ss = SpreadsheetApp.getActive();
  var shC = ss.getSheetByName(HOJA.CONCILIACION);
  var datosC = shC && shC.getLastRow() > 1 ? shC.getRange(2, 1, shC.getLastRow() - 1, COLS.length).getValues() : [];
  var cambiosC = false, aplicados = 0;
  var cuentaProv = cfg.cuentaProveedores || 2102001;
  var plan = null;
  var flujoUpd = [];

  for (var k in porMov) {
    var mv = M.porClave[k], x = porMov[k];
    if (!mv) continue;
    var docs = x.docs.map(function (r) {
      var t = parseInt(r[FC['TIPO DOC']], 10);
      return { abrev: SII_ABREV[t] || 'DOC', folio: r[FC['FOLIO']], razonSocial: r[FC['RAZON SOCIAL']], rut: r[FC['RUT PROVEEDOR']] };
    });
    var confirmado = x.estados.every(function (e) { return e === 'CONCILIADA' || e === 'CONFIRMADA'; });
    var resumen = docs.map(function (d) { return d.abrev + '.' + d.folio; }).join(', ') + ' ' + docs[0].razonSocial + ' (RUT ' + docs[0].rut + ')';
    if (mv.origen === 'CONCILIACION') {
      var r = datosC[mv.fila - 2];
      if (!r) continue;
      var expl = String(r[C['EXPLICACIÓN']] || '').replace(/^\[SII\][^|]*\|\s*/, '');
      if (confirmado) {
        if (!plan) plan = leerPlanCuentasCache_();
        if (r[C['ESTADO']] !== ESTADO.MANUAL) {
          r[C['ESTADO']] = ESTADO.IDENTIFICADO;
          r[C['FUENTE']] = 'FACTURA SII';
          r[C['CONFIANZA']] = Math.max(mtNum(x.docs[0][FC['CONFIANZA']]), 0.9);
          r[C['CUENTA CONTABLE']] = cuentaProv;
          r[C['NOMBRE CUENTA']] = plan[cuentaProv] || 'Proveedores Nacionales';
          r[C['GLOSA']] = siiGlosa(docs, x.cuotas);
          r[C['APROBAR']] = true;
          r[C['RUT']] = docs[0].rut;                    // el RUT de la factura manda sobre lo sugerido por el modelo
          r[C['RAZON SOCIAL']] = docs[0].razonSocial;
        } else {                                        // fila editada por el analista: solo completar lo vacío
          if (!r[C['GLOSA']]) r[C['GLOSA']] = siiGlosa(docs, x.cuotas);
          if (!r[C['RUT']]) r[C['RUT']] = docs[0].rut;
          if (!r[C['RAZON SOCIAL']]) r[C['RAZON SOCIAL']] = docs[0].razonSocial;
        }
        r[C['EXPLICACIÓN']] = '[SII] Pago de ' + resumen + ' | ' + expl;
        aplicados++;
      } else {
        r[C['EXPLICACIÓN']] = '[SII] Posible pago de ' + resumen + ' (confirmar en FACTURAS COMPRA) | ' + expl;
      }
      cambiosC = true;
    } else if (confirmado && (!mv.rut || !mv.razon)) {
      flujoUpd.push({ fila: mv.fila, rut: mv.rut || docs[0].rut, razon: mv.razon || docs[0].razonSocial });
    }
  }
  if (cambiosC) {
    shC.getRange(2, 1, datosC.length, COLS.length).setValues(datosC);
    colorear_(shC, 2, datosC.length);
  }
  if (flujoUpd.length) {
    var t = leerTabla_(HOJA.FLUJO);
    var iR = col_(t, 'RUT'), iZ = col_(t, 'RAZON SOCIAL');
    if (iR >= 0 && iZ >= 0) flujoUpd.forEach(function (u) {
      t.hoja.getRange(u.fila, iR + 1).setValue(u.rut);
      t.hoja.getRange(u.fila, iZ + 1).setValue(u.razon);
    });
  }
  return aplicados;
}

/* ================================================================== */
/* Edición en FACTURAS COMPRA                                          */
/* ================================================================== */

/** CONFIRMAR marcado -> CONFIRMADA (y todo su grupo). Desmarcado -> se rechaza ese movimiento. */
function onEditFacturas_(e) {
  var sh = e.range.getSheet();
  if (e.range.getRow() < 2 || e.range.getColumn() > FC['CONFIRMAR'] + 1 || e.range.getLastColumn() < FC['CONFIRMAR'] + 1) return;
  var n = sh.getLastRow() - 1;
  var filas = sh.getRange(2, 1, n, FCOLS.length).getValues();
  var col = coloresFacturas_();
  for (var i = e.range.getRow() - 2; i <= e.range.getLastRow() - 2; i++) {
    var r = filas[i];
    if (!r) continue;
    var clave = r[FC['CLAVE MOV']];
    var grupo = filas.filter(function (o) { return clave && o[FC['CLAVE MOV']] === clave; });
    if (r[FC['CONFIRMAR']] === true) {
      if (!clave) { r[FC['CONFIRMAR']] = false; r[FC['EXPLICACIÓN']] = 'No hay un pago propuesto para confirmar. ' + r[FC['EXPLICACIÓN']]; continue; }
      grupo.forEach(function (o) { o[FC['ESTADO']] = 'CONFIRMADA'; o[FC['CONFIRMAR']] = true; });
    } else if (clave && F_FIJOS.concat(['POR REVISAR']).indexOf(r[FC['ESTADO']]) >= 0) {
      revertirConciliacion_(r);
      grupo.forEach(function (o) {
        var rech = String(o[FC['MOVS RECHAZADOS']] || '').split(F_SEP).filter(String);
        String(clave).split(F_SEP).forEach(function (k) { if (rech.indexOf(k) < 0) rech.push(k); });
        o[FC['MOVS RECHAZADOS']] = rech.join(F_SEP);
        o[FC['ESTADO']] = 'SIN PAGO'; o[FC['CONFIRMAR']] = false; o[FC['CONFIANZA']] = 0;
        o[FC['EXPLICACIÓN']] = 'Pago propuesto rechazado por el analista. Use el menú 4 para buscar otro.';
      });
    }
  }
  sh.getRange(2, 1, n, FCOLS.length).setValues(filas);
  sh.getRange(2, 1, n, FCOLS.length).setBackgrounds(filas.map(function (r) { var c = col[r[FC['ESTADO']]] || '#ffffff'; return FCOLS.map(function () { return c; }); }));
}

/** Deshace en CONCILIACION lo que se había aplicado desde una factura cuyo cruce el analista rechazó. */
function revertirConciliacion_(r) {
  var ubic = String(r[FC['UBICACIÓN']] || '').split(F_SEP);
  var dets = String(r[FC['DETALLE TRANSFERENCIA']] || '').split(F_SEP);
  var sh = SpreadsheetApp.getActive().getSheetByName(HOJA.CONCILIACION);
  if (!sh) return;
  ubic.forEach(function (u, i) {
    var m = u.match(/^CONCILIACION fila (\d+)/);
    if (!m) return;
    var fila = +m[1];
    if (fila > sh.getLastRow()) return;
    var x = sh.getRange(fila, 1, 1, COLS.length).getValues()[0];
    if (String(x[C['Detalle Movimiento']]).trim() !== String(dets[i] || '').trim()) return;   // la fila cambió
    var expl = String(x[C['EXPLICACIÓN']] || '').replace(/^\[SII\][^|]*\|\s*/, '');
    if (x[C['FUENTE']] === 'FACTURA SII') {
      x[C['ESTADO']] = ESTADO.REVISAR; x[C['FUENTE']] = 'FACTURA RECHAZADA'; x[C['APROBAR']] = false;
      x[C['GLOSA']] = ''; x[C['RUT']] = ''; x[C['RAZON SOCIAL']] = '';
    }
    x[C['EXPLICACIÓN']] = '[SII] Se rechazó el cruce con ' + (SII_ABREV[parseInt(r[FC['TIPO DOC']], 10)] || 'DOC') + '.' + r[FC['FOLIO']] + ' ' +
      r[FC['RAZON SOCIAL']] + '; revise la clasificación | ' + expl;
    sh.getRange(fila, 1, 1, COLS.length).setValues([x]);
    colorear_(sh, fila, 1);
  });
}

function filtrarFacturasPendientes() {
  var sh = hojaFacturas_();
  if (sh.getLastRow() < 2) return;
  var f = sh.getFilter();
  if (f) f.remove();
  sh.getRange(1, 1, sh.getLastRow(), FCOLS.length).createFilter().setColumnFilterCriteria(FC['ESTADO'] + 1,
    SpreadsheetApp.newFilterCriteria().setHiddenValues(F_FIJOS).build());
  SpreadsheetApp.getActive().setActiveSheet(sh);
}
