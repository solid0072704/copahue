/**
 * Simulación de punta a punta de Codigo.gs con un SpreadsheetApp falso:
 * configuración → carga de cartola → edición manual → traspaso a FLUJO.
 *
 * Uso: [RCV=rcv1.csv,rcv2.csv] [DETALLE=1] node test/simulacion.js flujo.json cartola.(json|csv) [bbdd.json] [AAAA-MM]
 *   RCV: Registro(s) de Compras del SII a cargar después de la cartola (como en el diálogo).
 *   cartola.json: hoja exportada (se procesa como si se pegara en la hoja CARTOLA).
 *   cartola.csv:  archivo descargado del banco (se procesa como en el diálogo de carga,
 *                 con el mismo lector CSV de Dialogo.html).
 *   AAAA-MM: deja en FLUJO solo la historia anterior a ese mes (simula cartola nueva).
 */
var fs = require('fs'), vm = require('vm'), path = require('path');
var flujo = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
var esCsv = /\.csv$/i.test(process.argv[3]);
var cartola = esCsv ? leerCsvDialogo(fs.readFileSync(process.argv[3], 'utf8')) : JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

/** Usa la misma función leerCsv del diálogo, para probar exactamente lo que corre en el navegador. */
function leerCsvDialogo(texto) {
  var html = fs.readFileSync(path.join(__dirname, '..', 'Dialogo.html'), 'utf8');
  var js = html.slice(html.indexOf('  function leerCsv'), html.indexOf('  function detectarCodificacion'));
  var c = {}; vm.createContext(c); vm.runInContext(js, c);
  c.__t = texto;
  return JSON.parse(vm.runInContext('JSON.stringify(leerCsv(__t))', c));
}
var bbdd = process.argv[4] ? JSON.parse(fs.readFileSync(process.argv[4], 'utf8')) : null;
var corte = process.argv[5];

function aFecha(v) { return typeof v === 'string' && /^\d{4}-\d{2}-\d{2}T/.test(v) ? new Date(+v.slice(0, 4), +v.slice(5, 7) - 1, +v.slice(8, 10)) : v; }
function limpiar(m) { return m.map(function (r) { return r.map(function (c) { return c === null ? '' : aFecha(c); }); }); }
flujo = limpiar(flujo);
if (corte) flujo = [flujo[0]].concat(flujo.slice(1).filter(function (r) {
  return !(r[1] instanceof Date) || (r[1].getFullYear() * 100 + r[1].getMonth() + 1) < +corte.replace('-', '');
}));

/* ---------- SpreadsheetApp falso ---------- */
function Hoja(nombre, datos) { this.nombre = nombre; this.d = datos || []; this.filtro = null; }
Hoja.prototype = {
  getName: function () { return this.nombre; },
  ancho: function () { return this.d.reduce(function (m, r) { return Math.max(m, r.length); }, 0); },
  getLastRow: function () { for (var i = this.d.length; i > 0; i--) if (this.d[i - 1].some(function (c) { return c !== '' && c !== null; })) return i; return 0; },
  getLastColumn: function () { return this.ancho(); },
  getDataRange: function () { return this.getRange(1, 1, Math.max(1, this.getLastRow()), Math.max(1, this.ancho())); },
  getRange: function (r, c, nr, nc) {
    if (typeof r === 'string') {
      var m = r.match(/^([A-Z])(\d+)(?::([A-Z])(\d+))?$/);
      var c1 = m[1].charCodeAt(0) - 64, r1 = +m[2];
      return new Rango(this, r1, c1, m[4] ? +m[4] - r1 + 1 : 1, m[3] ? m[3].charCodeAt(0) - 64 - c1 + 1 : 1);
    }
    return new Rango(this, r, c, nr || 1, nc || 1);
  },
  appendRow: function (f) { this.d.splice(this.getLastRow(), 0, f.slice()); },
  setFrozenRows: function () {}, setFrozenColumns: function () {}, setColumnWidth: function () {},
  getFilter: function () { return this.filtro; },
  getActiveRange: function () { return this.getRange(2, 1); }
};
function Rango(h, r, c, nr, nc) { this.h = h; this.r = r; this.c = c; this.nr = nr; this.nc = nc; }
var noop = ['setNumberFormat', 'setDataValidation', 'setBackgrounds', 'setBackground', 'clearDataValidations', 'setWrap',
  'setFontColor', 'setNote', 'setFontWeight', 'setVerticalAlignment', 'check'];
noop.forEach(function (n) { Rango.prototype[n] = function () { return this; }; });
Rango.prototype.getRow = function () { return this.r; };
Rango.prototype.getColumn = function () { return this.c; };
Rango.prototype.getLastColumn = function () { return this.c + this.nc - 1; };
Rango.prototype.getNumRows = function () { return this.nr; };
Rango.prototype.getSheet = function () { return this.h; };
Rango.prototype.getValues = function () {
  var out = [];
  for (var i = 0; i < this.nr; i++) {
    var f = this.h.d[this.r - 1 + i] || [];
    var x = [];
    for (var j = 0; j < this.nc; j++) { var v = f[this.c - 1 + j]; x.push(v === undefined || v === null ? '' : v); }
    out.push(x);
  }
  return out;
};
Rango.prototype.setValues = function (v) {
  if (v.length !== this.nr || v[0].length !== this.nc) throw new Error('setValues: tamaño ' + v.length + 'x' + v[0].length + ' ≠ ' + this.nr + 'x' + this.nc);
  for (var i = 0; i < this.nr; i++) {
    var fi = this.r - 1 + i;
    while (this.h.d.length <= fi) this.h.d.push([]);
    for (var j = 0; j < this.nc; j++) this.h.d[fi][this.c - 1 + j] = v[i][j];
  }
  return this;
};
Rango.prototype.setValue = function (v) { return this.setValues([[v]]); };
Rango.prototype.getValue = function () { return this.getValues()[0][0]; };
Rango.prototype.getLastRow = function () { return this.r + this.nr - 1; };
Rango.prototype.clearContent = function () { return this.setValues(this.getValues().map(function (r) { return r.map(function () { return ''; }); })); };
Rango.prototype.insertCheckboxes = function () {
  var v = this.getValues().map(function (r) { return r.map(function (c) { return c === '' ? false : c; }); });
  return this.setValues(v);
};
Rango.prototype.createFilter = function () { var f = { setColumnFilterCriteria: function () { return f; }, remove: function () {} }; this.h.filtro = f; return f; };

var hojas = { 'FLUJO': new Hoja('FLUJO', flujo), 'CARTOLA': new Hoja('CARTOLA', esCsv ? [] : limpiar(cartola)) };
if (bbdd) hojas['BBDD'] = new Hoja('BBDD', limpiar(bbdd));
var alertas = [];
var ss = {
  getSheetByName: function (n) { return hojas[n] || null; },
  insertSheet: function (n) { return (hojas[n] = new Hoja(n, [])); },
  setActiveSheet: function () {}, getActiveSheet: function () { return hojas['CONCILIACION']; }
};
var dv = { requireValueInList: function () { return dv; }, setAllowInvalid: function () { return dv; }, build: function () { return {}; } };
var fc = { setHiddenValues: function () { return fc; }, build: function () { return {}; } };
var ui = {
  Button: { OK: 'OK', YES: 'YES' }, ButtonSet: { OK_CANCEL: 1, YES_NO: 2, OK: 3 },
  alert: function () { alertas.push([].slice.call(arguments).filter(function (a) { return typeof a === 'string'; }).join(' / ')); return 'YES'; },
  prompt: function () { return { getSelectedButton: function () { return 'OK'; }, getResponseText: function () { return ''; } }; }
};
var ctx = {
  console: console,
  SpreadsheetApp: { getActive: function () { return ss; }, getUi: function () { return ui; }, getActiveSheet: function () { return hojas['CONCILIACION']; },
    newDataValidation: function () { return dv; }, newFilterCriteria: function () { return fc; } },
  Session: { getScriptTimeZone: function () { return 'America/Santiago'; } },
  Utilities: { formatDate: function (d) { return d.toISOString().slice(0, 16).replace('T', ' '); } },
  CacheService: { getDocumentCache: function () { var m = {}; return { get: function (k) { return m[k] || null; }, put: function (k, v) { m[k] = v; }, remove: function (k) { delete m[k]; } }; } }
};
vm.createContext(ctx);
['Motor.gs', 'Cartola.gs', 'Sii.gs', 'Codigo.gs', 'Facturas.gs'].forEach(function (f) {
  vm.runInContext(fs.readFileSync(path.join(__dirname, '..', f), 'utf8'), ctx, { filename: f });
});

/* ---------- Escenario ---------- */
var filasFlujoAntes = hojas.FLUJO.getLastRow();
ctx.configurar();
function cargar() {
  if (!esCsv) return ctx.procesarHojaCartola();
  // Igual que el diálogo: analizar, aceptar la empresa sugerida de cada cuenta y procesar
  var nombre = path.basename(process.argv[3]);
  var an = ctx.analizarCartola(nombre, cartola);
  var empresas = {};
  an.bloques.forEach(function (b) {
    if (!b.n) return;
    empresas[b.i] = b.empresaSugerida;
    console.log('  cuenta ' + b.cuenta + ' ' + b.titular + ' (' + b.n + ' movs) → ' + (b.empresaSugerida || '(sin sugerencia: se omite)'));
  });
  var res = ctx.procesarCartolas([{ nombre: nombre, filas: cartola, empresas: empresas }]);
  alertas.push(vm.runInContext('textoResumen_', ctx)(res));
}
cargar();
var conc = hojas.CONCILIACION;
var C = vm.runInContext('C', ctx);
var filas = conc.getRange(2, 1, conc.getLastRow() - 1, 24).getValues();
var cnt = {};
filas.forEach(function (r) { cnt[r[0]] = (cnt[r[0]] || 0) + 1; });
console.log('CONCILIACION:', filas.length, 'filas', cnt);
if (process.env.DETALLE) filas.forEach(function (r) {
  console.log([r[C['ESTADO']], Math.round(r[C['CONFIANZA']] * 100) + '%', r[C['EMPRESA']], String(r[C['Detalle Movimiento']]).trim(),
    r[C['Cheque o Cargo']] ? '-' + r[C['Cheque o Cargo']] : '+' + r[C['Deposito o Abono']],
    '→ ' + r[C['CUENTA CONTABLE']] + ' ' + r[C['NOMBRE CUENTA']], '"' + r[C['GLOSA']] + '"', r[C['C.COSTO']]].join(' | '));
});

/* ---------- Facturas SII ---------- */
var FC = vm.runInContext('FC', ctx);
function verFacturas(titulo) {
  var sh = hojas['FACTURAS COMPRA'];
  if (!sh) return;
  var fs_ = sh.getRange(2, 1, sh.getLastRow() - 1, 26).getValues();
  var c = {};
  fs_.forEach(function (r) { c[r[0]] = (c[r[0]] || 0) + 1; });
  console.log(titulo, fs_.length, 'documentos', c);
  if (process.env.DETALLE_F) fs_.forEach(function (r) {
    console.log('  ' + [r[FC['ESTADO']], Math.round(r[FC['CONFIANZA']] * 100) + '%', r[FC['FOLIO']], r[FC['RAZON SOCIAL']], '$' + r[FC['MONTO A PAGAR']], '→',
      r[FC['FECHA PAGO']] instanceof Date || (r[FC['FECHA PAGO']] && r[FC['FECHA PAGO']].getTime) ? r[FC['FECHA PAGO']].toISOString().slice(0, 10) : '',
      r[FC['CUENTA DE PAGO']], '"' + r[FC['DETALLE TRANSFERENCIA']] + '"', r[FC['MONTO TRANSFERIDO']], r[FC['UBICACIÓN']],
      r[FC['PAGO CONJUNTO CON']] ? 'con ' + r[FC['PAGO CONJUNTO CON']] : ''].join(' | ').slice(0, 240));
  });
  return fs_;
}
if (process.env.RCV) {
  var rcv = process.env.RCV.split(',').map(function (p) {
    var filasR = leerCsvDialogo(fs.readFileSync(p, 'utf8'));
    var an = ctx.analizarRcv(path.basename(p), filasR);
    console.log('RCV ' + path.basename(p) + ': RUT ' + an.rutEmpresa + ' período ' + an.periodo + ', ' + an.n + ' docs (' + an.nuevos + ' nuevos), pagos se buscan en: ' +
      (an.empresasAsignadas.join(', ') || '(sin asignar → ' + (process.env.RCV_EMPRESA || '') + ')'));
    return { nombre: path.basename(p), filas: filasR, empresa: an.empresasAsignadas.length ? '' : (process.env.RCV_EMPRESA || '') };
  });
  var t0 = Date.now();
  var rr = ctx.procesarArchivos({ cartolas: [], rcv: rcv });
  console.log('procesarArchivos:', Date.now() - t0, 'ms; nuevos', rr.carga.nuevos, '| aplicados a CONCILIACION:', rr.facturas.aplicados);
  var ff = verFacturas('FACTURAS COMPRA:');
  var conSii = conc.getRange(2, 1, conc.getLastRow() - 1, 24).getValues().filter(function (r) { return r[C['FUENTE']] === 'FACTURA SII'; });
  console.log('Movimientos de CONCILIACION completados con factura:', conSii.length);
  conSii.slice(0, 6).forEach(function (r) {
    console.log('  ' + [String(r[C['Detalle Movimiento']]).trim(), '-' + r[C['Cheque o Cargo']], '→', r[C['CUENTA CONTABLE']], '"' + r[C['GLOSA']] + '"',
      r[C['RUT']], r[C['RAZON SOCIAL']], r[C['ESTADO']], r[C['APROBAR']]].join(' | '));
  });
  // El analista confirma un POR REVISAR y rechaza una CONCILIADA
  var shF = hojas['FACTURAS COMPRA'];
  var iRev = ff.findIndex(function (r) { return r[0] === 'POR REVISAR'; });
  if (iRev >= 0) {
    shF.getRange(iRev + 2, FC['CONFIRMAR'] + 1).setValue(true);
    ctx.onEdit({ range: shF.getRange(iRev + 2, FC['CONFIRMAR'] + 1) });
    console.log('Confirmada a mano fila', iRev + 2, '→', shF.getRange(iRev + 2, 1).getValue());
  }
  var cons = ff.map(function (r, i) { return i; }).filter(function (i) { return ff[i][0] === 'CONCILIADA' && String(ff[i][FC['UBICACIÓN']]).indexOf('CONCILIACION') === 0; });
  var iCon = cons.length > 1 ? cons[cons.length - 1] : -1;
  if (iCon >= 0) {
    var ubic = String(shF.getRange(iCon + 2, FC['UBICACIÓN'] + 1).getValue());
    var filaC = +ubic.replace(/\D+/g, ' ').trim().split(' ')[0];
    var antes = shF.getRange(iCon + 2, FC['DETALLE TRANSFERENCIA'] + 1).getValue();
    shF.getRange(iCon + 2, FC['CONFIRMAR'] + 1).setValue(false);
    ctx.onEdit({ range: shF.getRange(iCon + 2, FC['CONFIRMAR'] + 1) });
    ctx.cruzarFacturas();
    console.log('Rechazada fila', iCon + 2, '(era "' + antes + '") → ahora', shF.getRange(iCon + 2, 1).getValue(), '"' +
      shF.getRange(iCon + 2, FC['DETALLE TRANSFERENCIA'] + 1).getValue() + '"');
    var xc = conc.getRange(filaC, 1, 1, 24).getValues()[0];
    console.log('  CONCILIACION fila', filaC, 'quedó:', xc[C['ESTADO']], xc[C['FUENTE']], '| glosa "' + xc[C['GLOSA']] + '" | ' + String(xc[C['EXPLICACIÓN']]).slice(0, 110));
  }
}

// El analista clasifica el primer PENDIENTE a mano
var iPend = filas.findIndex(function (r) { return r[0] === 'PENDIENTE'; });
if (iPend >= 0) {
  var fila = iPend + 2;
  conc.getRange(fila, C['CUENTA CONTABLE'] + 1).setValue(2102001);
  ctx.onEdit({ range: conc.getRange(fila, C['CUENTA CONTABLE'] + 1) });
  var r = conc.getRange(fila, 1, 1, 24).getValues()[0];
  console.log('Edición manual fila', fila, '→', r[C['ESTADO']], '| nombre:', r[C['NOMBRE CUENTA']], '| aprobar:', r[C['APROBAR']]);
}

ctx.traspasarAFlujo();
var agregadas = hojas.FLUJO.getLastRow() - filasFlujoAntes;
if (process.env.RCV) {
  ctx.cruzarFacturas();
  var ub = hojas['FACTURAS COMPRA'].getRange(2, FC['UBICACIÓN'] + 1, hojas['FACTURAS COMPRA'].getLastRow() - 1, 1).getValues()
    .map(function (r) { return String(r[0]).split(' ')[0]; });
  var cu = {}; ub.forEach(function (u) { cu[u || '(sin pago)'] = (cu[u || '(sin pago)'] || 0) + 1; });
  console.log('Tras traspasar, ubicación de los pagos de las facturas:', cu);
}
console.log('Filas agregadas a FLUJO:', agregadas, '| quedan en CONCILIACION:', Math.max(0, conc.getLastRow() - 1));
var ult = hojas.FLUJO.d[hojas.FLUJO.getLastRow() - 1];
console.log('Última fila FLUJO:', flujo[0].map(function (h, i) { return h + '=' + (ult[i] instanceof Date ? ult[i].toISOString().slice(0, 10) : ult[i]); }).join(' | '));

// Reprocesar la misma cartola: ahora todo lo traspasado debe salir como YA EN FLUJO
cargar();
var f2 = conc.getRange(2, 1, conc.getLastRow() - 1, 24).getValues();
var cnt2 = {};
f2.forEach(function (r) { cnt2[r[0]] = (cnt2[r[0]] || 0) + 1; });
console.log('Reproceso de la misma cartola:', cnt2);
console.log('Alertas mostradas:\n - ' + alertas.join('\n - '));
