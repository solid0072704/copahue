/**
 * Cartola.gs — Lectura de cartolas bancarias (Banco de Chile, Santander y
 * formatos genéricos con columnas Fecha / Detalle / Cargo / Abono / Saldo).
 *
 * Recibe la cartola como matriz de celdas (tal cual sale de la planilla) y
 * devuelve los movimientos normalizados. Código puro, testeable en Node.
 */

var CARTOLA_COLUMNAS = [
  // [campo, expresión regular del encabezado]   (el orden importa)
  ['fecha',    /^fecha( (de )?(operaci[oó]n|contable|movimiento|transacci[oó]n))?$/],
  ['detalle',  /detalle|descripci[oó]n|glosa|concepto|^movimiento/],
  ['abono',    /abono|dep[oó]sito|cr[eé]dito|ingreso/],
  ['cargo',    /cargo|d[eé]bito|giro|egreso|cheque/],
  ['monto',    /^monto|^importe/],
  ['saldo',    /saldo/],
  ['docto',    /docto|documento|n[°º]? ?doc|^n[°º]$|n[uú]mero/],
  ['trn',      /^trn/],
  ['caja',     /^caja/],
  ['sucursal', /sucursal|oficina|canal/]
];

function crNorm_(s) {
  return String(s === null || s === undefined ? '' : s).toLowerCase().replace(/\s+/g, ' ').trim();
}

function crMapearColumnas_(enc) {
  var mapa = {}, usadas = {};
  for (var k = 0; k < CARTOLA_COLUMNAS.length; k++) {
    var campo = CARTOLA_COLUMNAS[k][0], re = CARTOLA_COLUMNAS[k][1];
    for (var j = 0; j < enc.length; j++) {
      if (usadas[j]) continue;
      var c = crNorm_(enc[j]);
      if (c && re.test(c)) { mapa[campo] = j; usadas[j] = 1; break; }
    }
  }
  return mapa;
}

var CR_CP1252_ = { 0x20AC: 0x80, 0x201A: 0x82, 0x0192: 0x83, 0x201E: 0x84, 0x2026: 0x85, 0x2020: 0x86, 0x2021: 0x87,
  0x02C6: 0x88, 0x2030: 0x89, 0x0160: 0x8A, 0x2039: 0x8B, 0x0152: 0x8C, 0x017D: 0x8E, 0x2018: 0x91, 0x2019: 0x92,
  0x201C: 0x93, 0x201D: 0x94, 0x2022: 0x95, 0x2013: 0x96, 0x2014: 0x97, 0x02DC: 0x98, 0x2122: 0x99, 0x0161: 0x9A,
  0x203A: 0x9B, 0x0153: 0x9C, 0x017E: 0x9E, 0x0178: 0x9F };

/** Repara texto UTF-8 leído como Latin-1/Windows-1252 ("AgrÃ­cola" -> "Agrícola"). */
function crReparar_(s) {
  s = String(s === null || s === undefined ? '' : s);
  if (!/[ÃÂ]/.test(s)) return s;
  var bytes = '';
  for (var i = 0; i < s.length; i++) {
    var c = s.charCodeAt(i);
    if (c > 255) { c = CR_CP1252_[c]; if (c === undefined) return s; }
    bytes += '%' + (c < 16 ? '0' : '') + c.toString(16);
  }
  try { return decodeURIComponent(bytes); } catch (e) { return s; }
}

/** ¿La fila es de encabezados de movimientos? */
function crEsEncabezado_(f) {
  var tieneFecha = false, tieneDet = false, tieneMonto = false;
  for (var j = 0; j < (f || []).length; j++) {
    var c = crNorm_(f[j]);
    if (!c) continue;
    if (/^fecha/.test(c)) tieneFecha = true;
    else if (/detalle|descripci|glosa|concepto|movimiento/.test(c)) tieneDet = true;
    if (/cargo|abono|monto|d[eé]bito|cr[eé]dito|dep[oó]sito|importe/.test(c)) tieneMonto = true;
  }
  return tieneFecha && tieneDet && tieneMonto;
}

/** Extrae titular / RUT / N° de cuenta de las líneas previas a un encabezado. */
function crDatosCuenta_(lineas, nombreArchivo) {
  var b = { titular: '', rut: '', cuenta: '', banco: '' };
  var txt = lineas.join(' ') + ' ' + (nombreArchivo || '');
  var m = txt.match(/(?:cta|cuenta|n[°º] ?de cuenta|cuenta corriente)[\s.:Nº°#]*([\d][\d\-. ]{5,})/i);
  if (m) b.cuenta = m[1].replace(/\D/g, '').replace(/^0+/, '');
  m = txt.match(/(\d{1,2}\.?\d{3}\.?\d{3}-[\dkK])/);
  if (m) b.rut = m[1].replace(/\./g, '');
  var primera = '';
  for (var i = lineas.length - 1; i >= 0; i--) if (String(lineas[i]).trim()) { primera = String(lineas[i]); break; }
  m = primera.match(/^\s*([^(]+?)\s*\(/);
  b.titular = m ? m[1].trim() : primera.trim().slice(0, 60);
  if (/banco de chile|edwards/i.test(txt)) b.banco = 'CHILE';
  else if (/santander/i.test(txt)) b.banco = 'SANTANDER';
  return b;
}

/**
 * Lee una cartola. Soporta archivos con UNA cuenta o con VARIAS cuentas
 * consecutivas (p.ej. la cartola consolidada del Banco de Chile, donde cada
 * cuenta trae su línea "Titular (RUT) cta:NNNN" y su fila de encabezados).
 *
 * @param {Array<Array>} filas  celdas de la cartola
 * @return {{bloques: Array<{titular, rut, cuenta, banco, movimientos, advertencias}>, advertencias: Array}}
 */
function parsearCartola(filas, nombreArchivo) {
  filas = (filas || []).map(function (r) { return (r || []).map(function (c) { return typeof c === 'string' ? crReparar_(c) : c; }); });
  var res = { bloques: [], advertencias: [] };
  var encabezados = [];
  for (var i = 0; i < filas.length; i++) if (crEsEncabezado_(filas[i])) encabezados.push(i);
  if (!encabezados.length) {
    res.advertencias.push('No se encontró la fila de encabezados (Fecha / Detalle / Cargo / Abono).');
    return res;
  }

  for (var k = 0; k < encabezados.length; k++) {
    var h = encabezados[k];
    var fin = k + 1 < encabezados.length ? encabezados[k + 1] : filas.length;
    var desdeCab = k === 0 ? 0 : encabezados[k - 1] + 1;
    // Líneas de cabecera de este bloque: filas sin fecha válida justo antes del encabezado
    var cab = [];
    for (var j = h - 1; j >= desdeCab && cab.length < 6; j--) {
      var fj = filas[j] || [];
      if (mtFecha(fj[0]) && fj.length > 2 && String(fj[1] || '').trim()) break;   // es un movimiento del bloque anterior
      cab.unshift(fj.filter(function (x) { return x !== '' && x !== null; }).join(' '));
    }
    var b = crDatosCuenta_(cab, nombreArchivo);
    b.movimientos = [];
    b.advertencias = [];

    var enc = filas[h];
    var c = crMapearColumnas_(enc);
    if (/cheque o cargo/.test(enc.map(crNorm_).join('|'))) b.banco = 'CHILE';
    if (c.fecha === undefined || c.detalle === undefined) { b.advertencias.push('No se reconocieron las columnas Fecha y Detalle.'); res.bloques.push(b); continue; }
    if (c.cargo === undefined && c.abono === undefined && c.monto === undefined) { b.advertencias.push('No se reconocieron columnas de Cargo/Abono/Monto.'); res.bloques.push(b); continue; }

    // Fin real del bloque: antes de las líneas de cabecera del bloque siguiente
    var omitidas = 0;
    for (i = h + 1; i < fin; i++) {
      var f = filas[i] || [];
      var fecha = mtFecha(f[c.fecha]);
      var detalle = f[c.detalle];
      if (!fecha) {
        if (k + 1 === encabezados.length && String(f.join('')).trim()) omitidas++;
        continue;
      }
      if (/^\s*saldo (inicial|anterior|final)/i.test(String(detalle))) continue;
      var cargo = c.cargo !== undefined ? Math.abs(mtNum(f[c.cargo])) : 0;
      var abono = c.abono !== undefined ? Math.abs(mtNum(f[c.abono])) : 0;
      if (c.monto !== undefined && c.cargo === undefined && c.abono === undefined) {
        var mo = mtNum(f[c.monto]);
        if (mo < 0) cargo = -mo; else abono = mo;
      }
      if (!cargo && !abono) continue;
      b.movimientos.push({
        fecha: fecha,
        detalle: String(detalle === null || detalle === undefined ? '' : detalle),
        cargo: cargo, abono: abono,
        saldo: c.saldo !== undefined ? mtNum(f[c.saldo]) : '',
        docto: c.docto !== undefined ? f[c.docto] : '',
        trn: c.trn !== undefined ? f[c.trn] : '',
        caja: c.caja !== undefined ? f[c.caja] : '',
        sucursal: c.sucursal !== undefined ? f[c.sucursal] : ''
      });
    }
    if (omitidas) b.advertencias.push(omitidas + ' filas sin fecha válida fueron omitidas (totales, notas, etc.).');
    res.bloques.push(b);
  }
  var total = res.bloques.reduce(function (s, b) { return s + b.movimientos.length; }, 0);
  if (!total) res.advertencias.push('La cartola no tiene movimientos.');
  return res;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { parsearCartola: parsearCartola };
}
