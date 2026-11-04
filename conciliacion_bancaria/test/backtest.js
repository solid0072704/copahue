/**
 * Backtest del motor contra la historia real (fuera de Apps Script).
 *
 * Uso:  node test/backtest.js flujo.json [meses=4]
 *   flujo.json = hoja FLUJO exportada como arreglo de filas (primera fila = encabezados).
 *
 * Para cada uno de los últimos N meses entrena con todo lo anterior y
 * clasifica los movimientos del mes, comparando contra lo que hizo el analista.
 */
var fs = require('fs');
var M = require('../Motor.gs');

var archivo = process.argv[2];
var nMeses = +(process.argv[3] || 4);
var filas = JSON.parse(fs.readFileSync(archivo, 'utf8'));
var enc = filas[0].map(function (h) { return String(h || '').trim().toUpperCase(); });
function col(n) { var i = enc.indexOf(n); if (i < 0) throw new Error('Falta columna ' + n); return i; }
var C = { emp: col('EMPRESA'), fecha: col('FECHA'), det: col('DETALLE MOVIMIENTO'), cargo: col('CHEQUE O CARGO'),
  abono: col('DEPOSITO O ABONO'), d1: col('DETALLE 1'), cc: col('C.COSTO'), glosa: col('GLOSA'),
  cta: col('CUENTA CONTABLE'), ncta: col('NOMBRE CUENTA'), centro: col('CENTRO COSTO'), rut: col('RUT'), razon: col('RAZON SOCIAL') };

var hist = filas.slice(1).map(function (r) {
  return { empresa: r[C.emp], fecha: r[C.fecha], detalle: r[C.det], cargo: r[C.cargo], abono: r[C.abono],
    detalle1: r[C.d1], ccosto: r[C.cc], glosa: r[C.glosa], cuenta: r[C.cta], nombreCuenta: r[C.ncta],
    centro: r[C.centro], rut: r[C.rut], razon: r[C.razon] };
}).filter(function (h) { return h.fecha && h.cuenta && !/saldo caja inicial/i.test(h.detalle); });

var meses = {};
hist.forEach(function (h) { meses[M.mtClaveFecha(h.fecha).slice(0, 7)] = 1; });
var lista = Object.keys(meses).sort().slice(-nMeses);

var global = {};
lista.forEach(function (mes) {
  var train = hist.filter(function (h) { return M.mtClaveFecha(h.fecha).slice(0, 7) < mes; });
  var test = hist.filter(function (h) { return M.mtClaveFecha(h.fecha).slice(0, 7) === mes; });
  var modelo = new M.ModeloConciliacion(train, {});
  var b = {};
  test.forEach(function (h) {
    var p = modelo.predecir(h);
    var e = p.estado;
    var x = b[e] = b[e] || { n: 0, cta: 0, full: 0 };
    var g = global[e] = global[e] || { n: 0, cta: 0, full: 0 };
    var okC = p.label && String(p.label.cuenta) === String(h.cuenta);
    var okF = okC && String(p.label.ccosto) === String(h.ccosto || '') && String(p.label.centro) === String(h.centro || '');
    x.n++; g.n++; if (okC) { x.cta++; g.cta++; } if (okF) { x.full++; g.full++; }
  });
  console.log('\n== Mes ' + mes + ' (entrenado con ' + train.length + ' movs, prueba ' + test.length + ')');
  imprimir(b, test.length);
});
console.log('\n== TOTAL');
imprimir(global, Object.keys(global).reduce(function (s, k) { return s + global[k].n; }, 0));

function imprimir(b, total) {
  Object.keys(b).forEach(function (e) {
    var x = b[e];
    console.log('  ' + (e + '            ').slice(0, 13) + ' ' + pct(x.n, total) + ' de los movs | cuenta correcta ' +
      pct(x.cta, x.n) + ' | cuenta+ccosto correctos ' + pct(x.full, x.n));
  });
}
function pct(a, b) { return ((b ? a / b : 0) * 100).toFixed(1).padStart(5) + '%'; }
