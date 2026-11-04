/**
 * Prueba del cruce facturas SII ↔ pagos contra la historia real.
 *
 * Uso: node test/prueba_facturas.js flujo.json documentos.(csv|json) "EMPRESA 1,EMPRESA 2" [AAAA-MM-DD]
 *   documentos.csv  = RCV_COMPRA_REGISTRO_<RUT>_<AAAAMM>.csv descargado del SII
 *   documentos.json = hoja "COMPRAS SII" exportada (mismas columnas que el RCV)
 *   AAAA-MM-DD      = (opcional) los alias se aprenden solo con la historia anterior a esta fecha
 *
 * Para medir sin "hacer trampa", las glosas de los movimientos se borran antes de
 * cruzar; después se compara con lo que registró el analista (glosa "Pago FC.<folio> ...").
 */
var fs = require('fs'), vm = require('vm'), path = require('path');
var ctx = { console: console };
vm.createContext(ctx);
['Motor.gs', 'Cartola.gs', 'Sii.gs'].forEach(function (f) {
  vm.runInContext(fs.readFileSync(path.join(__dirname, '..', f), 'utf8'), ctx, { filename: f });
});
var flujo = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
var archivo = process.argv[3];
var filasDocs = /\.csv$/i.test(archivo)
  ? fs.readFileSync(archivo, 'utf8').replace(/^﻿/, '').replace(/\r/g, '').split('\n').map(function (l) { return l.split(';'); })
  : JSON.parse(fs.readFileSync(archivo, 'utf8'));
ctx.__a = { flujo: flujo, filas: filasDocs, nombre: path.basename(archivo), empresas: (process.argv[4] || '').split(',').filter(String),
            corte: process.argv[5] || '' };

var r = vm.runInContext('(' + function () {
  var A = __a;
  var enc = A.flujo[0].map(function (h) { return mtNormTexto(h); });
  function c(n) { return enc.indexOf(mtNormTexto(n)); }
  var rows = A.flujo.slice(1);
  var movs = [], hist = [];
  rows.forEach(function (r, i) {
    var cargo = mtNum(r[c('Cheque o Cargo')]);
    if (!cargo || !r[c('Fecha')]) return;
    var mv = { clave: 'F' + (i + 2), origen: 'FLUJO', fila: i + 2, empresa: r[c('EMPRESA')], fecha: r[c('Fecha')],
      detalle: r[c('Detalle Movimiento')], cargo: cargo, glosaReal: r[c('GLOSA')], glosa: '', rut: '' };
    movs.push(mv);
    if (!A.corte || String(r[c('Fecha')]).slice(0, 10) < A.corte)
      hist.push({ detalle: mv.detalle, cargo: cargo, glosa: r[c('GLOSA')], rut: r[c('RUT')], razon: r[c('RAZON SOCIAL')] });
  });
  var info = parsearRcv(A.filas, A.nombre);
  var docs = info.documentos.map(function (d, i) { d.id = 'D' + i; d.empresas = A.empresas; return d; });
  var cruce = new CruceFacturas(movs, hist, {});
  var t0 = Date.now();
  var res = cruce.cruzar(docs, {});
  var ms = Date.now() - t0;

  // Verdad: movimientos cuya glosa (registrada por el analista) menciona el folio
  var porEstado = {}, ejemplos = [];
  docs.forEach(function (d) {
    var x = res[d.id];
    var verdad = movs.filter(function (m) { return siiGlosaTieneFolio(m.glosaReal, d.folio) &&
      (Math.abs(m.cargo - d.montoPagar) <= 1 || siiSimNombre(siiContraparte(m.detalle), siiTokensRazon(d.razonSocial)) >= 0.5 ||
       siiSimNombre(siiTokensRazon(String(m.glosaReal).replace(/^.*?\d\s*/, '')), siiTokensRazon(d.razonSocial)) >= 0.5); })
      .map(function (m) { return m.clave; });
    var e = porEstado[x.estado] = porEstado[x.estado] || { n: 0, conVerdad: 0, correctos: 0 };
    e.n++;
    if (verdad.length) {
      e.conVerdad++;
      var ok = x.movs.length && x.movs.every(function (m) { return verdad.indexOf(m.clave) >= 0; });
      if (ok) e.correctos++;
      if (!ok && ejemplos.length < 15) ejemplos.push(d.abrev + '.' + d.folio + ' ' + d.razonSocial + ' $' + d.montoPagar + ' [' + x.estado + '] → ' +
        (x.movs.map(function (m) { return m.fila + ' "' + String(m.detalle).trim() + '" $' + m.cargo; }).join(' + ') || '—') +
        ' | real: ' + verdad.map(function (v) { var m = movs.filter(function (z) { return z.clave === v; })[0]; return m.fila + ' "' + String(m.detalle).trim() + '" $' + m.cargo; }).join(' + '));
    }
  });
  var detalle = docs.slice(0, 40).map(function (d) {
    var x = res[d.id];
    return [x.estado, Math.round(x.confianza * 100) + '%', d.abrev + '.' + d.folio, d.razonSocial, '$' + d.montoPagar, '→',
      x.movs.map(function (m) { return mtClaveFecha(m.fecha) + ' ' + m.empresa + ' "' + String(m.detalle).trim() + '" $' + m.cargo; }).join(' + '),
      '|', x.explicacion].join(' ');
  });
  return { rutEmpresa: info.rutEmpresa, periodo: info.periodo, documentos: docs.length, ms: ms, porEstado: porEstado, errores: ejemplos, detalle: detalle };
} + ')()', ctx);
console.log(JSON.stringify(r, null, 1));
