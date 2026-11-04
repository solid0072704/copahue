/**
 * Prueba de punta a punta (sin Google): lee una cartola y la clasifica
 * contra la historia, igual que lo hace el aplicativo.
 *
 * Uso: node test/prueba_cartola.js flujo.json cartola.json [AAAA-MM]
 *   flujo.json / cartola.json = hojas exportadas como arreglos de filas.
 *   La empresa de cada cuenta de la cartola se adivina por el nombre del titular.
 *   AAAA-MM (opcional) = entrenar solo con la historia anterior a ese mes,
 *   para simular que la cartola es nueva.
 */
var fs = require('fs');
var vm = require('vm');
var path = require('path');

var ctx = { console: console };
vm.createContext(ctx);
['Motor.gs', 'Cartola.gs', 'Codigo.gs'].forEach(function (f) {
  vm.runInContext(fs.readFileSync(path.join(__dirname, '..', f), 'utf8'), ctx, { filename: f });
});

var flujo = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
var cartola = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
var corte = process.argv[4];

ctx.__flujo = flujo; ctx.__cartola = cartola; ctx.__corte = corte || '';
var r = vm.runInContext('(' + function () {
  var enc = __flujo[0].map(function (h) { return mtNormTexto(h); });
  function c(n) { return enc.indexOf(mtNormTexto(n)); }
  var hist = __flujo.slice(1).map(function (r) {
    return { empresa: r[c('EMPRESA')], fecha: r[c('Fecha')], detalle: r[c('Detalle Movimiento')], cargo: r[c('Cheque o Cargo')],
      abono: r[c('Deposito o Abono')], detalle1: r[c('DETALLE 1')], ccosto: r[c('C.COSTO')], glosa: r[c('GLOSA')],
      cuenta: r[c('CUENTA CONTABLE')], nombreCuenta: r[c('NOMBRE CUENTA')], centro: r[c('CENTRO COSTO')], rut: '', razon: '' };
  }).filter(function (h) { return h.fecha && (!__corte || mtClaveFecha(h.fecha).slice(0, 7) < __corte); });

  var info = parsearCartola(__cartola, 'cartola.xls');
  var modelo = new ModeloConciliacion(hist, {});
  var existentes = {};
  hist.forEach(function (h) {
    var k = mtClaveDuplicado(h.empresa, h.fecha, h.detalle, h.cargo, h.abono);
    existentes[k] = (existentes[k] || 0) + 1;
  });
  var cuenta = {}, ejemplos = [], empresas = {}, asignacion = [];
  hist.forEach(function (h) { if (h.empresa) empresas[String(h.empresa).trim()] = 1; });
  empresas = Object.keys(empresas);
  var movs = [];
  info.bloques.forEach(function (b) {
    if (!b.movimientos.length) return;
    var e = adivinarEmpresa_(b, empresas, {}, {}).empresa;
    asignacion.push(b.cuenta + ' ' + b.titular + ' -> ' + (e || '(sin empresa, se omite)') + ' [' + b.movimientos.length + ']');
    if (!e) return;
    b.movimientos.forEach(function (m) { m.empresa = e; movs.push(m); });
  });
  movs.forEach(function (m) {
    var p = clasificarMovimiento_(m, modelo, [], [], {}, {}, existentes);
    cuenta[p.estado] = (cuenta[p.estado] || 0) + 1;
    if (ejemplos.length < 12) ejemplos.push([p.estado, p.confianza, m.detalle.trim(), m.cargo || -m.abono, p.label ? p.label.cuenta + ' ' + p.label.glosa : '—'].join(' | '));
  });
  // Fila que se agregaría a FLUJO
  var fila = [];
  COLS.forEach(function (n, i) { fila[i] = ''; });
  var m0 = movs[0];
  fila[C['EMPRESA']] = m0.empresa; fila[C['Fecha']] = m0.fecha; fila[C['Detalle Movimiento']] = m0.detalle;
  fila[C['Cheque o Cargo']] = m0.cargo; fila[C['Deposito o Abono']] = m0.abono; fila[C['CUENTA CONTABLE']] = 2102001;
  fila[C['GLOSA']] = 'prueba';
  var ff = filaFlujo_(fila, enc, { 2102001: 'PROVEEDORES' }, { PROVEEDORES: { cat2: 'Flujo Operativo', cat3: 'Proveedores' } });
  return { cuentas: asignacion, movimientos: movs.length, porEstado: cuenta, ejemplos: ejemplos,
           filaFlujo: __flujo[0].map(function (h, i) { return h + '=' + (ff[i] instanceof Date ? mtClaveFecha(ff[i]) : ff[i]); }) };
}.toString() + ')()', ctx);

console.log(JSON.stringify(r, null, 1));
