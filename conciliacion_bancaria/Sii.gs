/**
 * Sii.gs — Registro de Compras del SII (RCV) y cruce de facturas con pagos.
 *
 * 1. parsearRcv(): lee el CSV "RCV_COMPRA_REGISTRO_<RUT>_<AAAAMM>.csv" que se
 *    descarga del SII (separado por ;).
 * 2. CruceFacturas: propone qué movimiento bancario (cargo) paga cada factura:
 *      - monto exacto (una factura = una transferencia),
 *      - varias facturas del mismo proveedor pagadas en una transferencia
 *        (incluye notas de crédito que descuentan),
 *      - una factura pagada en varias transferencias (cuotas / abonos),
 *    y usa el nombre del beneficiario del detalle bancario, el RUT si aparece,
 *    los alias aprendidos de la historia (p.ej. "Traspaso A:cerda Sofia" pagó
 *    facturas de "DISEÑO GRÁFICO FOLLOWUS LIMITADA") y la cercanía de fechas.
 *
 * Código puro (sin servicios de Google), testeable en Node.
 */

var SII_TIPOS = {
  29: 'Factura de inicio', 30: 'Factura', 32: 'Factura exenta', 33: 'Factura electrónica',
  34: 'Factura exenta electrónica', 40: 'Liquidación factura', 43: 'Liquidación factura electrónica',
  45: 'Factura de compra', 46: 'Factura de compra electrónica', 55: 'Nota de débito',
  56: 'Nota de débito electrónica', 60: 'Nota de crédito', 61: 'Nota de crédito electrónica',
  914: 'Declaración de ingreso (DIN)'
};
var SII_ABREV = { 29: 'FC', 30: 'FC', 32: 'FC', 33: 'FC', 34: 'FC', 40: 'LF', 43: 'LF', 45: 'FC', 46: 'FC',
                  55: 'ND', 56: 'ND', 60: 'NC', 61: 'NC', 914: 'DIN' };

var CRUCE_DEFAULTS = {
  umbralFactura: 0.85,       // desde aquí el cruce queda CONCILIADA automáticamente
  umbralFacturaRevisar: 0.50,
  diasAntes: 30,             // un pago puede ser hasta 30 días anterior a la factura (anticipo)
  diasDespues: 180,          // ... y hasta 180 días posterior
  maxDocsGrupo: 6,
  maxCuotas: 4,
  buscarOtrasEmpresas: true, // si no aparece en la empresa, buscar en las demás del grupo (queda POR REVISAR)
  factorOtraEmpresa: 0.82
};

function siiEsNC(tipo) { return +tipo === 60 || +tipo === 61; }

/* ------------------------------------------------------------------ */
/* Lectura del RCV                                                     */
/* ------------------------------------------------------------------ */

/**
 * @param {Array<Array>} filas  celdas del CSV
 * @param {string} nombreArchivo  p.ej. RCV_COMPRA_REGISTRO_77450092-8_202609.csv
 * @return {{rutEmpresa, periodo, libro, estadoRcv, documentos:Array, advertencias:Array}}
 */
function parsearRcv(filas, nombreArchivo) {
  var res = { rutEmpresa: '', periodo: '', libro: 'COMPRA', estadoRcv: '', documentos: [], advertencias: [] };
  var nom = String(nombreArchivo || '');
  var m = nom.match(/(\d{7,8}-[\dkK])/);
  if (m) res.rutEmpresa = m[1].toUpperCase();
  m = nom.match(/_(\d{4})(\d{2})(?=\D|$)/);
  if (m) res.periodo = m[1] + '-' + m[2];
  if (/venta/i.test(nom)) res.libro = 'VENTA';
  m = nom.match(/RCV_(?:COMPRA|VENTA)_([A-Z_]+?)_\d/i);
  if (m) res.estadoRcv = m[1].toUpperCase();

  var h = -1, enc = null;
  for (var i = 0; i < Math.min(filas.length, 10); i++) {
    var t = (filas[i] || []).map(function (c) { return mtNormTexto(c); });
    if (t.indexOf('folio') >= 0 && (t.indexOf('rut proveedor') >= 0 || t.indexOf('rut cliente') >= 0)) { h = i; enc = t; break; }
  }
  if (h < 0) { res.advertencias.push('No se encontró el encabezado del Registro de Compras (Folio / RUT Proveedor).'); return res; }
  if (enc.indexOf('rut cliente') >= 0) res.libro = 'VENTA';
  if (res.libro === 'VENTA') {
    res.advertencias.push('Es un Registro de VENTAS: por ahora el cruce solo trabaja con facturas de compra (pagos a proveedores).');
    return res;
  }
  function c(n) { return enc.indexOf(n); }
  var C = {
    tipo: c('tipo doc'), tipoCompra: c('tipo compra'), rut: c('rut proveedor'), razon: c('razon social'), folio: c('folio'),
    fecha: c('fecha docto'), recepcion: c('fecha recepcion'), exento: c('monto exento'), neto: c('monto neto'),
    ivaRec: c('monto iva recuperable'), ivaNoRec: c('monto iva no recuperable'), total: c('monto total'),
    otroImp: c('valor otro impuesto')
  };
  if (C.folio < 0 || C.rut < 0 || C.total < 0) { res.advertencias.push('Faltan columnas Folio / RUT Proveedor / Monto Total.'); return res; }

  for (i = h + 1; i < filas.length; i++) {
    var f = filas[i] || [];
    var folio = String(f[C.folio] === undefined ? '' : f[C.folio]).trim();
    var rut = String(f[C.rut] || '').trim().toUpperCase();
    if (!folio || !rut) continue;
    var tipo = +mtNum(f[C.tipo]);
    var total = mtNum(f[C.total]);
    var doc = {
      tipoDoc: tipo, tipoNombre: SII_TIPOS[tipo] || ('Tipo ' + tipo), abrev: SII_ABREV[tipo] || 'DOC',
      tipoCompra: C.tipoCompra >= 0 ? f[C.tipoCompra] : '',
      rutProveedor: rut, razonSocial: typeof crReparar_ === 'function' ? crReparar_(f[C.razon]) : String(f[C.razon] || ''),
      folio: folio.replace(/^0+(?=\d)/, ''),
      fechaDocto: mtFecha(f[C.fecha]), fechaRecepcion: C.recepcion >= 0 ? mtFecha(f[C.recepcion]) : null,
      exento: C.exento >= 0 ? mtNum(f[C.exento]) : 0, neto: C.neto >= 0 ? mtNum(f[C.neto]) : 0,
      iva: (C.ivaRec >= 0 ? mtNum(f[C.ivaRec]) : 0) + (C.ivaNoRec >= 0 ? mtNum(f[C.ivaNoRec]) : 0),
      total: Math.abs(total)
    };
    // Factura de compra (46/45): el comprador retiene el IVA, se paga el neto (+ exento)
    doc.montoPagar = (tipo === 46 || tipo === 45) ? doc.neto + doc.exento : doc.total;
    if (siiEsNC(tipo)) doc.montoPagar = -doc.montoPagar;
    res.documentos.push(doc);
  }
  if (!res.documentos.length) res.advertencias.push('El archivo no tiene documentos.');
  return res;
}

function siiClaveDoc(rutEmpresa, doc) {
  return [String(rutEmpresa).toUpperCase(), doc.tipoDoc, String(doc.rutProveedor).toUpperCase(), doc.folio].join('|');
}

/* ------------------------------------------------------------------ */
/* Nombres                                                             */
/* ------------------------------------------------------------------ */

var SII_STOP_OPERACION = ['traspaso', 'traspasos', 'transferencia', 'transf', 'transfer', 'pago', 'pagos', 'pac', 'cargo',
  'internet', 'en', 'linea', 'a', 'de', 'desde', 'hacia', 'al', 'cuenta', 'otro', 'otros', 'banco', 'bancos', 'via', 'spav',
  'tef', 'automatico', 'autom', 'mismo', 'dia', 'proveedores', 'provision'];
var SII_STOP_LEGAL = ['spa', 'ltda', 'limitada', 'sa', 's', 'eirl', 'e', 'i', 'r', 'l', 'y', 'cia', 'compania', 'de', 'del',
  'la', 'las', 'los', 'el', 'sociedad', 'comercial', 'servicios', 'servicio', 'inversiones', 'empresa', 'empresas',
  'chile', 'group', 'grupo', 'ltd', 'inc', 'sociedades', 'sucursal', 'asesorias', 'y'];

function siiTokens_(s, stops) {
  var t = mtNormTexto(typeof crReparar_ === 'function' ? crReparar_(s) : s).split(' ');
  var out = [];
  for (var i = 0; i < t.length; i++) {
    var w = t[i];
    if (!w || w.length < 2 || /^\d+$/.test(w) || stops.indexOf(w) >= 0 || SII_STOP_LEGAL.indexOf(w) >= 0) continue;
    if (out.indexOf(w) < 0) out.push(w);
  }
  return out;
}

/** Nombre de la contraparte en el detalle bancario ("Traspaso A:cerda Sofia" -> [cerda, sofia]). */
function siiContraparte(detalle) { return siiTokens_(detalle, SII_STOP_OPERACION); }
function siiTokensRazon(razon) { return siiTokens_(razon, []); }

function siiCoincide_(x, lista) {
  for (var i = 0; i < lista.length; i++) {
    var t = lista[i];
    if (t === x) return true;
    // el banco corta el nombre a 40 caracteres: "agropec" ~ "agropecuaria"
    if (x.length >= 3 && t.indexOf(x) === 0) return true;
    if (t.length >= 4 && x.indexOf(t) === 0) return true;
  }
  return false;
}

/** Similitud 0..1 entre el nombre del detalle y la razón social. */
function siiSimNombre(tokDetalle, tokRazon) {
  if (!tokDetalle.length || !tokRazon.length) return 0;
  var wa = 0, ma = 0, wb = 0, mb = 0, i;
  for (i = 0; i < tokDetalle.length; i++) {
    var w = Math.min(tokDetalle[i].length, 8);
    wa += w;
    if (siiCoincide_(tokDetalle[i], tokRazon)) ma += w;
  }
  for (i = 0; i < tokRazon.length; i++) {
    var w2 = Math.min(tokRazon[i].length, 8);
    wb += w2;
    if (siiCoincide_(tokRazon[i], tokDetalle)) mb += w2;
  }
  if (!ma) return 0;
  return 0.7 * (ma / wa) + 0.3 * (mb / wb);
}

function siiRutSinDv(rut) { return String(rut || '').replace(/[^\dkK]/g, '').slice(0, -1).replace(/^0+/, ''); }

/** ¿La glosa ya menciona este folio? ("Pago FC.69 ...", "Pago  Factura 46.946.561 ...") */
function siiGlosaTieneFolio(glosa, folio) {
  if (!glosa || !folio) return false;
  var g = String(glosa).replace(/(\d)\.(?=\d)/g, '$1');
  var re = new RegExp('(?:\\bfc|\\bfact\\w*|\\bf|\\bbh|\\bnc|\\bnd|\\bfolio|n[°º])\\s*\\.?\\s*(?:n[°º]?\\s*)?0*' + String(folio).replace(/\D/g, '') + '(?!\\d)', 'i');
  return re.test(g);
}

/* ------------------------------------------------------------------ */
/* Cruce                                                               */
/* ------------------------------------------------------------------ */

/**
 * @param {Array<Object>} movimientos  cargos candidatos:
 *   {clave, origen ('FLUJO'|'CONCILIACION'), fila, empresa, fecha, detalle, cargo, glosa, rut}
 * @param {Array<Object>} historia  filas de FLUJO {detalle, cargo, glosa, rut, razon} para aprender alias
 * @param {Object} opciones  ver CRUCE_DEFAULTS
 */
function CruceFacturas(movimientos, historia, opciones) {
  this.op = {};
  for (var k in CRUCE_DEFAULTS) this.op[k] = CRUCE_DEFAULTS[k];
  for (k in (opciones || {})) if (opciones[k] !== undefined && opciones[k] !== null && opciones[k] !== '') this.op[k] = opciones[k];
  this.movs = [];
  for (var i = 0; i < movimientos.length; i++) {
    var m = movimientos[i];
    var cargo = Math.abs(mtNum(m.cargo));
    var f = mtFecha(m.fecha);
    if (!cargo || !f) continue;
    var x = {};
    for (k in m) x[k] = m[k];
    x.cargo = cargo; x.fechaD = f; x.t = f.getTime();
    x.tokens = siiContraparte(m.detalle);
    x.empresaN = mtNormEmpresa(m.empresa);
    x.numeros = (String(m.detalle || '').replace(/[.\-]/g, '').match(/\d{6,}/g) || []).map(function (n) { return n.replace(/^0+/, ''); });
    // Traspasos entre cuentas propias y fondos mutuos: nunca se proponen solo por coincidir el monto
    x.interno = /cuenta\s*:?\s*0*\d{6,}|\bfmu\b|fondos? mutuos?|inversiones 0?965712208|rescate/i.test(String(m.detalle || ''));
    this.movs.push(x);
  }
  // Alias aprendidos: nombre en el banco -> razones sociales / RUT de facturas pagadas antes
  this.alias = {};
  for (i = 0; i < (historia || []).length; i++) {
    var h = historia[i];
    if (!mtNum(h.cargo)) continue;
    var tk = siiContraparte(h.detalle);
    if (!tk.length) continue;
    var key = tk.slice().sort().join(' ');
    var razon = '';
    var g = String(h.glosa || '').replace(/(\d)\.(?=\d)/g, '$1');
    var mm = g.match(/\b(?:FC|BH|NC|Factura|Fact)\b\.?\s*N?[°º]?\s*\d+\s*[,\-]?\s*(.+)$/i);
    if (h.razon) razon = String(h.razon);
    else if (mm) razon = mm[1];
    if (!razon && !h.rut) continue;
    var a = this.alias[key] = this.alias[key] || { razones: {}, ruts: {} };
    if (razon) a.razones[razon] = (a.razones[razon] || 0) + 1;
    if (h.rut) a.ruts[siiRutSinDv(h.rut)] = 1;
  }
}

/** Similitud del movimiento con el proveedor (0..1) y de dónde sale. */
CruceFacturas.prototype.simProveedor = function (mov, doc) {
  var rutN = siiRutSinDv(doc.rutProveedor);
  if (rutN && (mov.numeros.indexOf(rutN) >= 0 || siiRutSinDv(mov.rut) === rutN)) return { s: 1, por: 'RUT del proveedor en el movimiento' };
  var tr = doc._tokRazon || (doc._tokRazon = siiTokensRazon(doc.razonSocial));
  var s = siiSimNombre(mov.tokens, tr), por = 'nombre';
  var a = this.alias[mov.tokens.slice().sort().join(' ')];
  if (a) {
    if (rutN && a.ruts[rutN]) return { s: 0.97, por: 'antes se pagó a este proveedor con el mismo detalle bancario' };
    for (var r in a.razones) {
      var sa = siiSimNombre(siiTokensRazon(r), tr) * 0.95;
      if (sa > s) { s = sa; por = 'alias aprendido: "' + String(mov.detalle).trim() + '" pagó antes a ' + r; }
    }
  }
  return { s: s, por: por };
};

CruceFacturas.prototype.factorFecha_ = function (mov, doc) {
  if (!doc.fechaDocto) return 1;
  var d = (mov.t - doc.fechaDocto.getTime()) / 86400000;
  if (d < -this.op.diasAntes || d > this.op.diasDespues) return 0;
  if (d < -3) return 0.85 - Math.min(0.1, -d / 600);   // pago anticipado
  if (d <= 60) return 1 - d / 3000;                     // leve preferencia por el pago más cercano
  return 0.98 - Math.min(0.15, (d - 60) / 800);
};

/**
 * Cruza las facturas pendientes con los movimientos.
 * @param {Array<Object>} docs  [{id, empresas:[...], ...doc de parsearRcv}]
 * @param {Object} clavesUsadas  {claveMov: true} movimientos ya asignados (no se proponen)
 * @return {Object} id -> {estado, confianza, movs:[mov], docsGrupo:[id], tipo, explicacion}
 */
CruceFacturas.prototype.cruzar = function (docs, clavesUsadas) {
  var op = this.op, self = this;
  clavesUsadas = clavesUsadas || {};
  var opciones = [];
  var porProveedor = {};
  docs.forEach(function (d) {
    d._emps = (d.empresas || []).map(mtNormEmpresa);
    var k = d.rutProveedor + '§' + d._emps.join(',');
    (porProveedor[k] = porProveedor[k] || []).push(d);
  });

  function movsDe(d) {
    return self.movs.filter(function (m) {
      return !clavesUsadas[m.clave] && (!d._emps.length || op.buscarOtrasEmpresas || d._emps.indexOf(m.empresaN) >= 0) && self.factorFecha_(m, d) > 0;
    });
  }
  // Pago desde otra empresa del grupo: se permite, pero con menos confianza
  function fe(m, d) { return !d._emps.length || d._emps.indexOf(m.empresaN) >= 0 ? 1 : op.factorOtraEmpresa; }

  docs.forEach(function (d) {
    if (d.montoPagar <= 0) return;                  // las NC solo participan en grupos
    var cand = movsDe(d);
    var sims = cand.map(function (m) { return self.simProveedor(m, d); });
    var exactos = [];
    cand.forEach(function (m, i) {
      var reg = siiGlosaTieneFolio(m.glosa, d.folio);
      var sp = sims[i];
      if (reg && sp.s < 0.9) sp = { s: 0.9, por: 'la glosa ya menciona el folio' };
      // (a) una factura = una transferencia, monto exacto
      if (Math.abs(m.cargo - d.montoPagar) <= 1 && (sp.s >= 0.5 || !m.interno)) exactos.push({ m: m, sp: sp, reg: reg });
      // (b) una transferencia paga varias facturas del mismo proveedor
      if (sp.s >= 0.6 && m.cargo > d.montoPagar + 1) {
        var hermanos = (porProveedor[d.rutProveedor + '§' + d._emps.join(',')] || []).filter(function (o) {
          return o !== d && o.fechaDocto && Math.abs(m.t - o.fechaDocto.getTime()) / 86400000 <= op.diasDespues &&
                 (m.t - o.fechaDocto.getTime()) / 86400000 >= -op.diasAntes;
        }).slice(0, 14);
        var sub = siiSubconjunto_(hermanos.map(function (o) { return o.montoPagar; }), m.cargo - d.montoPagar, op.maxDocsGrupo - 1, 1);
        if (sub) {
          var grupo = [d].concat(sub.map(function (ix) { return hermanos[ix]; }));
          opciones.push({ docs: grupo, movs: [m], conf: Math.min(0.97, (0.75 + 0.2 * sp.s) * self.factorFecha_(m, d)) * fe(m, d), tipo: 'GRUPO', sp: sp,
                          reg: reg || grupo.some(function (o) { return siiGlosaTieneFolio(m.glosa, o.folio); }) });
        }
      }
    });
    exactos.forEach(function (x) {
      var ff = self.factorFecha_(x.m, d) * fe(x.m, d), conf;
      if (x.sp.s >= 0.5) conf = (0.80 + 0.19 * x.sp.s) * ff;
      else {
        // solo coincide el monto: aceptable si es el único movimiento con ese monto y no hay otra factura igual
        var otrasIguales = docs.filter(function (o) { return o !== d && Math.abs(o.montoPagar - d.montoPagar) <= 1 && o._emps.join() === d._emps.join(); }).length;
        conf = (exactos.length === 1 && !otrasIguales ? 0.6 : 0.45) * ff;
        x.sp = { s: x.sp.s, por: 'solo coincide el monto' + (exactos.length > 1 ? ' (hay ' + exactos.length + ' movimientos con ese monto)' : '') };
      }
      opciones.push({ docs: [d], movs: [x.m], conf: conf, tipo: 'EXACTO', sp: x.sp, reg: x.reg });
    });
    // (c) una factura pagada en varias transferencias al mismo proveedor
    var parciales = cand.map(function (m, i) { return { m: m, sp: sims[i] }; }).filter(function (x) {
      return (x.sp.s >= 0.6 || siiGlosaTieneFolio(x.m.glosa, d.folio)) && x.m.cargo < d.montoPagar - 1;
    }).slice(0, 14);
    var subc = siiSubconjunto_(parciales.map(function (x) { return x.m.cargo; }), d.montoPagar, op.maxCuotas, 2);
    if (subc && subc.length >= 2) {
      var ms = subc.map(function (ix) { return parciales[ix]; });
      var sMin = Math.min.apply(null, ms.map(function (x) { return x.sp.s; }));
      opciones.push({ docs: [d], movs: ms.map(function (x) { return x.m; }), conf: Math.min(0.95, 0.72 + 0.2 * sMin) * fe(ms[0].m, d), tipo: 'CUOTAS', sp: ms[0].sp,
                      reg: ms.some(function (x) { return siiGlosaTieneFolio(x.m.glosa, d.folio); }) });
    }
  });

  // Asignación: mejores opciones primero; cada factura y cada movimiento se usa una sola vez
  opciones.sort(function (a, b) {
    return (b.conf + (b.reg ? 0.05 : 0)) - (a.conf + (a.reg ? 0.05 : 0)) || a.movs.length - b.movs.length;
  });
  var docUsado = {}, movUsado = {}, res = {};
  opciones.forEach(function (o) {
    if (o.docs.some(function (d) { return docUsado[d.id]; }) || o.movs.some(function (m) { return movUsado[m.clave]; })) return;
    if (o.conf < op.umbralFacturaRevisar * 0.9) return;
    o.docs.forEach(function (d) { docUsado[d.id] = 1; });
    o.movs.forEach(function (m) { movUsado[m.clave] = 1; });
    var estado = o.reg && o.movs.every(function (m) { return m.origen === 'FLUJO'; }) ? 'YA REGISTRADA'
               : o.conf >= op.umbralFactura ? 'CONCILIADA' : o.conf >= op.umbralFacturaRevisar ? 'POR REVISAR' : 'POR REVISAR';
    o.docs.forEach(function (d) {
      res[d.id] = {
        estado: estado, confianza: Math.round(o.conf * 1000) / 1000, tipo: o.tipo, movs: o.movs,
        docsGrupo: o.docs.filter(function (x) { return x !== d; }).map(function (x) { return x.id; }),
        explicacion: siiExplicar_(o, d)
      };
    });
  });

  // Sin pago: dejar pistas (transferencias al mismo proveedor por otros montos)
  docs.forEach(function (d) {
    if (res[d.id]) return;
    var pistas = movsDe(d).filter(function (m) { return !movUsado[m.clave] && self.simProveedor(m, d).s >= 0.6; })
      .slice(0, 3).map(function (m) { return mtClaveFecha(m.fechaD) + ' $' + siiFmt_(m.cargo) + ' "' + String(m.detalle).trim() + '"'; });
    res[d.id] = { estado: siiEsNC(d.tipoDoc) ? 'NC SIN APLICAR' : 'SIN PAGO', confianza: 0, tipo: '', movs: [], docsGrupo: [],
      explicacion: (siiEsNC(d.tipoDoc) ? 'Nota de crédito: se descontará cuando aparezca un pago conjunto con otras facturas del proveedor.'
                    : 'No se encontró un cargo por $' + siiFmt_(d.montoPagar) + ' en las cuentas de la empresa.') +
                   (pistas.length ? ' Transferencias al mismo proveedor por otros montos: ' + pistas.join('; ') + '.' : '') };
  });
  return res;
};

function siiFmt_(n) { return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.'); }

function siiExplicar_(o, d) {
  var m = o.movs[0];
  var t = o.tipo === 'EXACTO' ? 'Cargo por el monto exacto ($' + siiFmt_(m.cargo) + ')'
        : o.tipo === 'GRUPO' ? 'Una transferencia de $' + siiFmt_(m.cargo) + ' paga ' + o.docs.length + ' documentos: ' +
                               o.docs.map(function (x) { return x.abrev + '.' + x.folio + ' $' + siiFmt_(x.montoPagar); }).join(' + ')
        : 'Pagada en ' + o.movs.length + ' transferencias: ' + o.movs.map(function (x) { return mtClaveFecha(x.fechaD) + ' $' + siiFmt_(x.cargo); }).join(' + ');
  var otra = d._emps && d._emps.length && o.movs.some(function (x) { return d._emps.indexOf(x.empresaN) < 0; });
  return t + '; ' + o.sp.por + (o.reg ? '; la glosa ya menciona el folio' : '') +
         (otra ? '; OJO: pagada desde otra empresa (' + o.movs[0].empresa + ')' : '') + '.';
}

/**
 * Busca índices de 'valores' que sumen 'objetivo' (±tolerancia), con entre
 * 'minimo' y 'maximo' elementos. Devuelve el primero encontrado (menos elementos primero).
 */
function siiSubconjunto_(valores, objetivo, maximo, minimo) {
  minimo = minimo || 1;
  var n = valores.length, mejor = null, pasos = 0;
  for (var k = minimo; k <= Math.min(maximo, n) && !mejor; k++) {
    (function busca(ini, elegidos, suma) {
      if (mejor || ++pasos > 20000) return;
      if (elegidos.length === k) {
        if (Math.abs(suma - objetivo) <= Math.max(1, k)) mejor = elegidos.slice();
        return;
      }
      for (var i = ini; i < n; i++) {
        elegidos.push(i);
        busca(i + 1, elegidos, suma + valores[i]);
        elegidos.pop();
        if (mejor) return;
      }
    })(0, [], 0);
  }
  return mejor;
}

/** Glosa al estilo de la empresa: "Pago FC.69 COMPRAVENTA FRESIA SPA" */
function siiGlosa(docs, parcial) {
  var d = docs[0];
  return 'Pago ' + docs.map(function (x) { return x.abrev + '.' + x.folio; }).join(', ') + ' ' + d.razonSocial + (parcial ? ' (pago parcial)' : '');
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { parsearRcv: parsearRcv, CruceFacturas: CruceFacturas, siiSimNombre: siiSimNombre, siiContraparte: siiContraparte,
    siiTokensRazon: siiTokensRazon, siiGlosaTieneFolio: siiGlosaTieneFolio, siiGlosa: siiGlosa, siiClaveDoc: siiClaveDoc };
}
