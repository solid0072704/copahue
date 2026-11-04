/**
 * Motor.gs — Motor predictivo de clasificación de movimientos bancarios.
 *
 * Código JavaScript puro (sin servicios de Google) para que pueda probarse
 * también fuera de Apps Script (ver test/backtest.js).
 *
 * Idea general (k vecinos más cercanos sobre la historia):
 *   1. Cada movimiento ya clasificado en la hoja FLUJO es un "ejemplo".
 *   2. Para un movimiento nuevo se buscan los ejemplos más parecidos:
 *        - similitud del texto del detalle (tokens ponderados por IDF,
 *          coincidencia exacta y coincidencia con números enmascarados),
 *        - misma empresa/cuenta bancaria,
 *        - mismo sentido (cargo / abono),
 *        - monto parecido,
 *        - antigüedad (lo reciente pesa un poco más).
 *   3. Los vecinos "votan" por una clasificación (cuenta contable + c.costo
 *      + centro de costo). La confianza combina el consenso de los votos con
 *      la similitud del mejor vecino.
 *   4. Según la confianza el movimiento queda IDENTIFICADO, POR REVISAR o
 *      PENDIENTE (para que el analista lo clasifique manualmente).
 *
 * Además se aplican, con prioridad, las REGLAS manuales y los cruces con
 * la hoja PAGOS POR HACER (monto exacto + nombre del beneficiario).
 */

var ESTADO = {
  IDENTIFICADO: 'IDENTIFICADO',
  REVISAR: 'POR REVISAR',
  PENDIENTE: 'PENDIENTE',
  DUPLICADO: 'YA EN FLUJO',
  MANUAL: 'MANUAL'
};

var MOTOR_DEFAULTS = {
  umbralIdentificado: 0.80,
  umbralRevisar: 0.55,
  vecinos: 12,
  similitudMinima: 0.30,
  pesoOtraEmpresa: 0.70,
  diasPagoVentana: 60
};

/* ------------------------------------------------------------------ */
/* Utilidades de texto                                                 */
/* ------------------------------------------------------------------ */

function mtQuitarTildes_(s) {
  return String(s).normalize('NFD').replace(/[̀-ͯ]/g, '');
}

function mtNormTexto(s) {
  if (s === null || s === undefined) return '';
  return mtQuitarTildes_(String(s).toLowerCase())
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Texto normalizado con los números reemplazados por '#'. */
function mtMascara(s) {
  return mtNormTexto(s).replace(/\d+/g, '#');
}

/** Tokens útiles: palabras de 2+ letras y números de 5+ dígitos (RUT, cuentas). */
function mtTokens(s) {
  var t = mtNormTexto(s).split(' ');
  var out = [];
  var vistos = {};
  for (var i = 0; i < t.length; i++) {
    var w = t[i];
    if (!w) continue;
    if (/^\d+$/.test(w)) {
      if (w.length < 5) continue;
      w = w.replace(/^0+/, '');       // 008090262110 == 8090262110
    } else if (w.length < 2) {
      continue;
    } else if (/\d/.test(w)) {
      w = w.replace(/\d+/g, '#');     // fc56952158 -> fc#
    }
    if (!vistos[w]) { vistos[w] = 1; out.push(w); }
  }
  return out;
}

function mtNormEmpresa(s) {
  return mtNormTexto(s).replace(/\s+/g, ' ');
}

function mtNum(v) {
  if (v === null || v === undefined || v === '') return 0;
  if (typeof v === 'number') return isFinite(v) ? v : 0;
  var s = String(v).trim();
  if (!s) return 0;
  var neg = /^\(.*\)$/.test(s) || /^-/.test(s) || /-$/.test(s);
  s = s.replace(/[^\d,.\-]/g, '').replace(/-/g, '');
  // Formato chileno: 1.234.567,89  -> 1234567.89
  if (s.indexOf(',') >= 0 && s.indexOf('.') >= 0) {
    s = s.lastIndexOf(',') > s.lastIndexOf('.') ? s.replace(/\./g, '').replace(',', '.') : s.replace(/,/g, '');
  } else if (s.indexOf(',') >= 0) {
    s = /,\d{1,2}$/.test(s) ? s.replace(',', '.') : s.replace(/,/g, '');
  } else if ((s.match(/\./g) || []).length > 1 || /\.\d{3}$/.test(s)) {
    s = s.replace(/\./g, '');
  }
  var n = parseFloat(s);
  if (!isFinite(n)) return 0;
  return neg ? -n : n;
}

function mtFecha(v) {
  if (Object.prototype.toString.call(v) === '[object Date]') return isNaN(v.getTime()) ? null : v;
  if (v === null || v === undefined || v === '') return null;
  if (typeof v === 'number') {             // número de serie de Excel
    if (v < 20000 || v > 80000) return null;
    var u = new Date(Math.round((v - 25569) * 86400000));
    return new Date(u.getUTCFullYear(), u.getUTCMonth(), u.getUTCDate());
  }
  var s = String(v).trim();
  var m = s.match(/^(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})/);   // dd/mm/aaaa
  if (m) {
    var y = +m[3]; if (y < 100) y += 2000;
    if (+m[2] < 1 || +m[2] > 12 || +m[1] < 1 || +m[1] > 31) return null;
    return new Date(y, +m[2] - 1, +m[1]);
  }
  m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);                         // aaaa-mm-dd
  if (m) return new Date(+m[1], +m[2] - 1, +m[3]);
  return null;   // otros textos (títulos, totales) no son fechas
}

function mtClaveFecha(d) {
  d = mtFecha(d);
  if (!d) return '';
  var m = d.getMonth() + 1, day = d.getDate();
  return d.getFullYear() + '-' + (m < 10 ? '0' : '') + m + '-' + (day < 10 ? '0' : '') + day;
}

/** Clave para detectar movimientos que ya están cargados en FLUJO. */
function mtClaveDuplicado(empresa, fecha, detalle, cargo, abono) {
  return [mtNormEmpresa(empresa), mtClaveFecha(fecha), mtNormTexto(detalle),
          Math.round(mtNum(cargo)), Math.round(mtNum(abono))].join('|');
}

/* ------------------------------------------------------------------ */
/* Modelo                                                              */
/* ------------------------------------------------------------------ */

/**
 * @param {Array<Object>} historia  ejemplos clasificados:
 *   {empresa, fecha, detalle, cargo, abono, detalle1, ccosto, glosa,
 *    cuenta, nombreCuenta, centro, rut, razon}
 * @param {Object} opciones  ver MOTOR_DEFAULTS
 */
function ModeloConciliacion(historia, opciones) {
  this.op = {};
  for (var k in MOTOR_DEFAULTS) this.op[k] = MOTOR_DEFAULTS[k];
  for (k in (opciones || {})) if (opciones[k] !== undefined && opciones[k] !== null && opciones[k] !== '') this.op[k] = opciones[k];

  this.ej = [];
  this.indice = {};       // token -> [idx]
  this.porTexto = {};     // texto normalizado -> [idx]
  this.porMascara = {};   // texto enmascarado -> [idx]
  this.df = {};
  this.fechaMax = 0;
  this.ccostoPorEmpresa = {};
  this.ccostoPorEmpresaCuenta = {};
  this.grupos = {};

  for (var i = 0; i < historia.length; i++) {
    var h = historia[i];
    if (!h || h.cuenta === '' || h.cuenta === null || h.cuenta === undefined) continue;
    if (/saldo caja inicial/i.test(String(h.detalle))) continue;
    var cargo = Math.abs(mtNum(h.cargo)), abono = Math.abs(mtNum(h.abono));
    if (!cargo && !abono) continue;
    var f = mtFecha(h.fecha);
    var e = {
      empresa: h.empresa, empresaN: mtNormEmpresa(h.empresa),
      fecha: f, t: f ? f.getTime() : 0,
      detalle: h.detalle, texto: mtNormTexto(h.detalle), mascara: mtMascara(h.detalle),
      tokens: mtTokens(h.detalle),
      sentido: abono > cargo ? 1 : -1, monto: Math.abs(abono - cargo) || Math.max(cargo, abono),
      label: {
        detalle1: h.detalle1 || '', ccosto: h.ccosto || '', glosa: h.glosa || '',
        cuenta: h.cuenta, nombreCuenta: h.nombreCuenta || '', centro: h.centro || '',
        rut: h.rut || '', razon: h.razon || ''
      }
    };
    // Se vota por la CUENTA CONTABLE; el centro de costo se elige después, dentro de la cuenta ganadora
    e.claveLabel = String(e.label.cuenta);
    var idx = this.ej.length;
    this.ej.push(e);
    if (e.t > this.fechaMax) this.fechaMax = e.t;
    (this.porTexto[e.empresaN + '§' + e.texto] = this.porTexto[e.empresaN + '§' + e.texto] || []).push(idx);
    (this.porMascara[e.mascara] = this.porMascara[e.mascara] || []).push(idx);
    for (var j = 0; j < e.tokens.length; j++) {
      var tk = e.tokens[j];
      (this.indice[tk] = this.indice[tk] || []).push(idx);
      this.df[tk] = (this.df[tk] || 0) + 1;
    }
    // Distribución "a priori" por empresa + sentido + tipo de operación (+ rango de monto)
    var gk = mtClavesGrupo_(e.empresaN, e.sentido, e.texto, e.monto);
    for (j = 0; j < gk.length; j++) {
      var g = this.grupos[gk[j]] = this.grupos[gk[j]] || { n: 0, labels: {} };
      g.n++;
      var gl = g.labels[e.claveLabel] = g.labels[e.claveLabel] || { n: 0, ultimo: idx };
      gl.n++; if (e.t >= this.ej[gl.ultimo].t) gl.ultimo = idx;
    }
    var cc = this.ccostoPorEmpresa[e.empresaN] = this.ccostoPorEmpresa[e.empresaN] || {};
    var ck = e.label.ccosto + '|' + e.label.centro;
    cc[ck] = (cc[ck] || 0) + 1;
    var kc = e.empresaN + '§' + e.claveLabel;
    var cc2 = this.ccostoPorEmpresaCuenta[kc] = this.ccostoPorEmpresaCuenta[kc] || {};
    cc2[ck] = (cc2[ck] || 0) + 1;
  }
  this.N = this.ej.length;
}

/**
 * Tipo de operación = primeras palabras del detalle antes del nombre de la
 * contraparte (p.ej. "traspaso de", "traspaso a", "pago servicio en internet").
 */
function mtTipoOperacion_(texto) {
  var t = texto.replace(/\d+/g, ' ').replace(/\s+/g, ' ').trim().split(' ');
  // "traspaso de cuenta 0088..." (entre cuentas propias) es distinto de "traspaso de juan perez"
  return t.slice(0, t[2] === 'cuenta' ? 3 : 2).join(' ');
}

function mtClavesGrupo_(empN, sentido, texto, monto) {
  var tipo = mtTipoOperacion_(texto);
  var rango = monto > 0 ? Math.floor(Math.log(monto) / Math.LN10 * 2) : 0;   // medias décadas
  return ['G2|' + empN + '|' + sentido + '|' + tipo + '|' + rango,
          'G1|' + empN + '|' + sentido + '|' + tipo];
}

/**
 * Predicción de respaldo cuando el detalle es nuevo (contraparte nunca vista):
 * usa la distribución histórica de la empresa para ese tipo de operación.
 */
ModeloConciliacion.prototype.predecirGrupo_ = function (empN, sentido, texto, monto) {
  var claves = mtClavesGrupo_(empN, sentido, texto, monto);
  for (var i = 0; i < claves.length; i++) {
    var g = this.grupos[claves[i]];
    if (!g || g.n < 5) continue;
    var mejor = null, mk = null;
    for (var k in g.labels) if (!mejor || g.labels[k].n > mejor.n) { mejor = g.labels[k]; mk = k; }
    var share = mejor.n / g.n;
    var soporte = Math.min(1, g.n / 20);
    // Techo 0.85: con mucho soporte y casi unanimidad puede quedar IDENTIFICADO
    var conf = Math.pow(share, 2) * soporte * (i === 0 ? 0.88 : 0.80);
    return { conf: conf, idx: mejor.ultimo, n: mejor.n, total: g.n, nivel: i };
  }
  return null;
};

ModeloConciliacion.prototype.idf = function (tk) {
  return Math.log(1 + this.N / ((this.df[tk] || 0) + 1));
};

ModeloConciliacion.prototype.simTokens = function (a, b) {
  if (!a.length || !b.length) return 0;
  var setB = {}, inter = 0, union = 0, i;
  for (i = 0; i < b.length; i++) setB[b[i]] = 1;
  var setA = {};
  for (i = 0; i < a.length; i++) {
    var w = this.idf(a[i]);
    setA[a[i]] = 1;
    union += w;
    if (setB[a[i]]) inter += w;
  }
  for (i = 0; i < b.length; i++) if (!setA[b[i]]) union += this.idf(b[i]);
  return union ? inter / union : 0;
};

/**
 * Predice la clasificación de un movimiento.
 * @param {Object} mov {empresa, fecha, detalle, cargo, abono}
 * @return {Object} {estado, confianza, fuente, label, vecinos, explicacion}
 */
ModeloConciliacion.prototype.predecir = function (mov) {
  var op = this.op;
  var cargo = Math.abs(mtNum(mov.cargo)), abono = Math.abs(mtNum(mov.abono));
  var sentido = abono > cargo ? 1 : -1;
  var monto = Math.abs(abono - cargo) || Math.max(cargo, abono);
  var empN = mtNormEmpresa(mov.empresa);
  var texto = mtNormTexto(mov.detalle), mascara = mtMascara(mov.detalle);
  var tokens = mtTokens(mov.detalle);
  var f = mtFecha(mov.fecha);
  var tRef = f ? f.getTime() : this.fechaMax;

  // 1. Candidatos (índice invertido + coincidencias exactas)
  var cand = {}, i, j, lista;
  lista = this.porTexto[empN + '§' + texto] || [];
  for (i = 0; i < lista.length; i++) cand[lista[i]] = 1;
  lista = this.porMascara[mascara] || [];
  for (i = 0; i < lista.length; i++) cand[lista[i]] = 1;
  var tkOrden = tokens.slice().sort(function (a, b) { return (this.df[a] || 0) - (this.df[b] || 0); }.bind(this));
  for (i = 0; i < tkOrden.length && i < 6; i++) {
    lista = this.indice[tkOrden[i]] || [];
    if (lista.length > 4000) continue;           // tokens demasiado comunes
    for (j = 0; j < lista.length; j++) cand[lista[j]] = 1;
  }

  // 2. Puntaje de cada candidato
  var puntajes = [];
  for (var key in cand) {
    var e = this.ej[key];
    if (e.sentido !== sentido) continue;
    var sim;
    if (e.texto === texto) sim = 1;
    else {
      sim = this.simTokens(tokens, e.tokens);
      if (e.mascara === mascara) sim = Math.max(sim, 0.92);
    }
    if (sim <= 0) continue;
    var mismaEmp = e.empresaN === empN;
    var dif = Math.abs(e.monto - monto) / Math.max(e.monto, monto, 1);
    var simMonto = dif < 0.005 ? 1 : Math.max(0, 1 - dif * 1.5);
    var edadDias = Math.max(0, (tRef - e.t) / 86400000);
    var recencia = 0.85 + 0.15 * Math.exp(-edadDias / 365);
    var s = sim * (mismaEmp ? 1 : op.pesoOtraEmpresa) * (0.85 + 0.15 * simMonto) * recencia;
    if (s >= op.similitudMinima) puntajes.push({ idx: +key, s: s, sim: sim, mismaEmp: mismaEmp, montoIgual: dif < 0.005 });
  }
  puntajes.sort(function (a, b) { return b.s - a.s; });
  var top = puntajes.slice(0, op.vecinos);
  var grupo = this.predecirGrupo_(empN, sentido, texto, monto);
  var self = this;

  // 3. Votación ponderada de los vecinos
  var ganador = null, conf = 0, base = null, explicacion = '', fuente = 'HISTORIA';
  if (top.length) {
    var votos = {}, total = 0;
    for (i = 0; i < top.length; i++) {
      var p = top[i], ej = this.ej[p.idx], w = p.s * p.s;
      var v = votos[ej.claveLabel] = votos[ej.claveLabel] || { w: 0, n: 0, mejor: p, mejorS: 0, mejorEmp: null };
      v.w += w; v.n++; total += w;
      if (p.s > v.mejorS) { v.mejorS = p.s; v.mejor = p; }
      if (p.mismaEmp && (!v.mejorEmp || p.s > v.mejorEmp.s)) v.mejorEmp = p;
    }
    var ganadorClave = null;
    for (var kl in votos) if (!ganador || votos[kl].w > ganador.w) { ganador = votos[kl]; ganadorClave = kl; }
    var share = ganador.w / total;
    var soporte = Math.min(1, 0.80 + 0.10 * ganador.n);
    conf = share * ganador.mejorS * soporte;
    // Centro de costo: votación entre los vecinos de la cuenta ganadora (los de la misma empresa pesan más)
    var ccVotos = {}, ccMejor = null;
    for (i = 0; i < top.length; i++) {
      var pe = this.ej[top[i].idx];
      if (pe.claveLabel !== ganadorClave) continue;
      var kcc = pe.label.ccosto + '|' + pe.label.centro;
      var vc = ccVotos[kcc] = ccVotos[kcc] || { w: 0, mejor: top[i] };
      // Recencia fuerte: si el equipo cambió la forma de asignar el centro de costo, manda lo último
      var edad = Math.max(0, (tRef - pe.t) / 86400000);
      vc.w += top[i].s * top[i].s * (top[i].mismaEmp ? 1 : 0.25) * Math.exp(-edad / 60);
      if (top[i].s > vc.mejor.s || (top[i].s === vc.mejor.s && pe.t > this.ej[vc.mejor.idx].t)) vc.mejor = top[i];
    }
    for (var kv in ccVotos) if (!ccMejor || ccVotos[kv].w > ccMejor.w) ccMejor = ccVotos[kv];
    // Ejemplo a copiar (glosa, etc.): el más parecido con esa cuenta y centro de costo
    base = this.ej[(ganador.mejorEmp && this.ej[ganador.mejorEmp.idx].label.ccosto + '|' + this.ej[ganador.mejorEmp.idx].label.centro ===
                    this.ej[ccMejor.mejor.idx].label.ccosto + '|' + this.ej[ccMejor.mejor.idx].label.centro ? ganador.mejorEmp : ccMejor.mejor).idx];
    explicacion = ganador.n + ' de ' + top.length + ' movimientos parecidos usan esta cuenta; similitud del mejor: ' +
                  Math.round(ganador.mejorS * 100) + '%' + (base.empresaN === empN ? '' : ' (otra empresa)') +
                  '. Ej.: ' + mtClaveFecha(base.fecha) + ' "' + String(base.detalle).trim() + '" → ' + base.label.glosa;
  }

  // 4. Respaldo estadístico por tipo de operación de la empresa
  if (grupo) {
    var baseG = this.ej[grupo.idx];
    if (base && baseG.claveLabel === base.claveLabel) {
      conf = 1 - (1 - conf) * (1 - grupo.conf * 0.5);           // ambos coinciden: refuerza
    } else if (grupo.conf > conf) {
      conf = grupo.conf; base = baseG; fuente = 'PATRÓN EMPRESA';
      explicacion = 'Contraparte nueva. En ' + baseG.empresa + ', ' + grupo.n + ' de ' + grupo.total +
                    ' movimientos del tipo "' + mtTipoOperacion_(texto) + '"' +
                    (grupo.nivel === 0 ? ' y monto similar' : '') + ' van a esta cuenta. Ej.: "' + baseG.label.glosa + '"';
    }
  }

  if (!base) {
    return { estado: ESTADO.PENDIENTE, confianza: 0, fuente: 'SIN HISTORIA', label: null, vecinos: [],
             explicacion: 'No hay movimientos parecidos en la historia.' };
  }

  var label = {};
  for (var lk in base.label) label[lk] = base.label[lk];
  if (base.empresaN !== empN) {
    // El ejemplo es de otra empresa: usar el centro de costo habitual de ESTA empresa
    var hc = this.ccostoHabitual(mov.empresa, label.cuenta);
    if (hc.ccosto || hc.centro) { label.ccosto = hc.ccosto; label.centro = hc.centro; }
  }

  var estado = conf >= op.umbralIdentificado ? ESTADO.IDENTIFICADO
             : conf >= op.umbralRevisar ? ESTADO.REVISAR : ESTADO.PENDIENTE;

  return {
    estado: estado,
    confianza: Math.round(conf * 1000) / 1000,
    fuente: fuente,
    label: label,
    vecinos: top.slice(0, 3).map(function (p) {
      var x = self.ej[p.idx];
      return { fecha: mtClaveFecha(x.fecha), empresa: x.empresa, detalle: String(x.detalle).trim(),
               monto: x.monto * x.sentido, glosa: x.label.glosa, cuenta: x.label.cuenta, s: Math.round(p.s * 100) / 100 };
    }),
    explicacion: explicacion
  };
};

/**
 * Centro de costo más usado por una empresa (si se indica cuenta, el más usado
 * por esa empresa en esa cuenta). Para clasificaciones que vienen de reglas,
 * pagos o ejemplos de otra empresa.
 */
ModeloConciliacion.prototype.ccostoHabitual = function (empresa, cuenta) {
  var empN = mtNormEmpresa(empresa);
  var cc = (cuenta !== undefined && cuenta !== '' && this.ccostoPorEmpresaCuenta[empN + '§' + cuenta]) || this.ccostoPorEmpresa[empN];
  if (!cc) return { ccosto: '', centro: '' };
  var mejor = null, n = -1;
  for (var k in cc) if (cc[k] > n) { n = cc[k]; mejor = k; }
  var p = mejor.split('|');
  return { ccosto: p[0], centro: p[1] };
};

/* ------------------------------------------------------------------ */
/* Reglas manuales                                                     */
/* ------------------------------------------------------------------ */

/**
 * reglas: [{activa, empresa, contiene, tipo, montoMin, montoMax, label:{...}}]
 * 'contiene' es texto (se busca normalizado) o /regex/.
 */
function mtAplicarReglas(reglas, mov) {
  var cargo = Math.abs(mtNum(mov.cargo)), abono = Math.abs(mtNum(mov.abono));
  var tipo = abono > cargo ? 'ABONO' : 'CARGO';
  var monto = Math.max(cargo, abono);
  var texto = mtNormTexto(mov.detalle);
  for (var i = 0; i < reglas.length; i++) {
    var r = reglas[i];
    if (!r.activa) continue;
    if (r.empresa && mtNormEmpresa(r.empresa) !== mtNormEmpresa(mov.empresa)) continue;
    if (r.tipo && String(r.tipo).toUpperCase() !== tipo) continue;
    if (r.montoMin !== '' && r.montoMin !== null && r.montoMin !== undefined && monto < mtNum(r.montoMin)) continue;
    if (r.montoMax !== '' && r.montoMax !== null && r.montoMax !== undefined && monto > mtNum(r.montoMax)) continue;
    var c = String(r.contiene || '').trim();
    if (!c) continue;
    var ok;
    var m = c.match(/^\/(.*)\/([a-z]*)$/);
    if (m) {
      try { ok = new RegExp(m[1], m[2] || 'i').test(String(mov.detalle)); } catch (err) { ok = false; }
    } else {
      ok = texto.indexOf(mtNormTexto(c)) >= 0;
    }
    if (ok) return { regla: i, label: r.label };
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Cruce con PAGOS POR HACER                                           */
/* ------------------------------------------------------------------ */

var MT_MESES = { ene: 0, feb: 1, mar: 2, abr: 3, may: 4, jun: 5, jul: 6, ago: 7, sep: 8, set: 8, oct: 9, nov: 10, dic: 11 };

/** Interpreta 'SEMANA 10 NOV 2025', '10/11/2025', fechas, etc. */
function mtFechaPago(v) {
  if (Object.prototype.toString.call(v) === '[object Date]') return v;
  var s = mtNormTexto(v);
  var m = s.match(/(\d{1,2}) ([a-z]{3})[a-z]* (\d{4})/);
  if (m && MT_MESES[m[2]] !== undefined) return new Date(+m[3], MT_MESES[m[2]], +m[1]);
  return mtFecha(v);
}

/**
 * pagos: [{fecha, origen, rut, nombre, monto, asunto, cuenta, nombreCuenta, usado}]
 * Busca un pago con el mismo monto y cuyo nombre aparezca en el detalle.
 */
function mtCruzarPago(pagos, mov, diasVentana) {
  var cargo = Math.abs(mtNum(mov.cargo)), abono = Math.abs(mtNum(mov.abono));
  if (!cargo || abono > cargo) return null;
  var tokDet = mtTokens(mov.detalle);
  var setDet = {};
  for (var i = 0; i < tokDet.length; i++) setDet[tokDet[i]] = 1;
  var f = mtFecha(mov.fecha);
  var mejor = null, mejorPts = 0;
  for (i = 0; i < pagos.length; i++) {
    var p = pagos[i];
    if (p.usado || Math.round(mtNum(p.monto)) !== Math.round(cargo)) continue;
    if (!p.cuenta) continue;
    if (f && p.fechaD && diasVentana) {
      var d = (f.getTime() - p.fechaD.getTime()) / 86400000;
      if (d < -10 || d > diasVentana) continue;
    }
    var tn = mtTokens(p.nombre), pts = 0;
    for (var j = 0; j < tn.length; j++) {
      if (tn[j].length < 3) continue;
      if (setDet[tn[j]]) pts++;
      else {  // detalle truncado por el banco (40 caracteres)
        for (var t in setDet) if (t.length >= 3 && tn[j].indexOf(t) === 0) { pts += 0.5; break; }
      }
    }
    var rutN = String(p.rut || '').replace(/[^\dkK]/g, '').slice(0, -1);
    if (rutN && setDet[rutN.replace(/^0+/, '')]) pts += 2;
    if (p.origen && mtNormEmpresa(mov.empresa).indexOf(mtNormEmpresa(p.origen)) === 0) pts += 0.5;
    if (pts > mejorPts) { mejorPts = pts; mejor = p; }
  }
  if (!mejor || mejorPts < 1) return null;
  return { pago: mejor, puntos: mejorPts };
}

/* ------------------------------------------------------------------ */
/* Traspasos entre cuentas propias                                     */
/* ------------------------------------------------------------------ */

/**
 * Índice N° de cuenta bancaria -> cuenta contable, a partir de los nombres del
 * plan de cuentas que incluyen el número (p.ej. 1101221 "Banco de Chile Los
 * Coihues 886-05150-09" -> 8860515009).
 * @param {Object} plan  {codigo: nombre}
 */
function mtIndiceCuentasPropias(plan) {
  var idx = {};
  for (var cod in plan) {
    var nom = String(plan[cod]);
    if (/no ocupar/i.test(nom)) continue;
    var nums = nom.match(/\d[\d\-. ]{5,}\d/g) || [];
    for (var i = 0; i < nums.length; i++) {
      var d = nums[i].replace(/\D/g, '').replace(/^0+/, '');
      if (d.length >= 7 && !idx[d]) idx[d] = { cuenta: /^\d+$/.test(cod) ? +cod : cod, nombre: nom };
    }
  }
  return idx;
}

/** Si el detalle menciona una cuenta propia ("Traspaso De Cuenta:008860515009"), la devuelve. */
function mtCuentaPropia(idx, detalle) {
  if (!idx) return null;
  var nums = String(detalle || '').match(/\d{7,}/g) || [];
  for (var i = 0; i < nums.length; i++) {
    var d = nums[i].replace(/^0+/, '');
    if (idx[d]) return { cuenta: idx[d].cuenta, nombre: idx[d].nombre, numero: d };
  }
  return null;
}

/* Exportación para pruebas en Node (no afecta a Apps Script). */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ModeloConciliacion: ModeloConciliacion, mtAplicarReglas: mtAplicarReglas, mtCruzarPago: mtCruzarPago,
    mtFechaPago: mtFechaPago, mtNormTexto: mtNormTexto, mtTokens: mtTokens, mtNum: mtNum, mtFecha: mtFecha,
    mtClaveDuplicado: mtClaveDuplicado, mtClaveFecha: mtClaveFecha, ESTADO: ESTADO, MOTOR_DEFAULTS: MOTOR_DEFAULTS,
    mtIndiceCuentasPropias: mtIndiceCuentasPropias, mtCuentaPropia: mtCuentaPropia };
}
