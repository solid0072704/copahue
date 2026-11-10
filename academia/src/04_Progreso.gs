/**
 * TPDV Academy · Avance de cada colaborador.
 */

/** Lecciones del usuario: { completadas: {idLeccion: true}, filas: {idLeccion: numeroDeFila} }. */
function leerProgresoUsuario_(email, filasPrecargadas) {
  var completadas = {};
  var filas = {};
  (filasPrecargadas || leerTabla_(HOJAS.PROGRESO)).forEach(function (p) {
    if (aTexto_(p['Email']).toLowerCase() !== email) return;
    var id = aTexto_(p['ID lección']);
    filas[id] = p._fila;
    if (esSi_(p['Completada'])) completadas[id] = true;
  });
  return { completadas: completadas, filas: filas };
}

/**
 * Resultados de evaluaciones por módulo:
 * { idModulo: { quiz: {mejor, aprobado, intentos, ultima}, practica: {puntaje, aprobado, fecha, evaluador, comentario} } }
 */
function leerResultadosUsuario_(email, filasPrecargadas) {
  var mapa = {};
  (filasPrecargadas || leerTabla_(HOJAS.RESULTADOS)).forEach(function (r) {
    if (aTexto_(r['Email']).toLowerCase() !== email) return;
    var id = aTexto_(r['ID módulo']);
    var tipo = aTexto_(r['Tipo']).toLowerCase();
    mapa[id] = mapa[id] || {};
    var puntaje = Number(r['Puntaje %']) || 0;
    var aprobado = esSi_(r['Aprobado']);
    var fecha = aFechaIso_(r['Fecha']);
    if (tipo.indexOf('pr') === 0) {
      // Evaluación práctica: vale el registro más reciente.
      var actual = mapa[id].practica;
      if (!actual || fecha >= actual.fecha) {
        mapa[id].practica = {
          puntaje: puntaje, aprobado: aprobado, fecha: fecha,
          evaluador: aTexto_(r['Evaluador']), comentario: aTexto_(r['Comentario'])
        };
      }
    } else {
      var q = mapa[id].quiz || { mejor: 0, aprobado: false, intentos: 0, ultima: '' };
      q.intentos += 1;
      q.mejor = Math.max(q.mejor, puntaje);
      q.aprobado = q.aprobado || aprobado;
      if (fecha > q.ultima) q.ultima = fecha;
      mapa[id].quiz = q;
    }
  });
  return mapa;
}

/**
 * Calcula el estado del programa para un usuario (función pura).
 * opciones: { ordenObligatorio: bool }
 */
function calcularEstado_(estructura, completadas, resultados, opciones) {
  var ordenObligatorio = !opciones || opciones.ordenObligatorio !== false;
  var unidades = 0;
  var unidadesHechas = 0;
  var leccionesHechas = 0;
  var anteriorCompleto = true;
  var siguiente = null;

  var modulos = estructura.modulos.map(function (m) {
    var res = resultados[m.id] || {};
    var obligatorias = m.lecciones.filter(function (l) { return l.obligatoria; });
    var lecciones = m.lecciones.map(function (l) {
      var hecha = !!completadas[l.id];
      if (hecha) leccionesHechas++;
      return Object.assign({}, l, { completada: hecha });
    });
    var hechas = obligatorias.filter(function (l) { return completadas[l.id]; }).length;
    var leccionesCompletas = hechas === obligatorias.length;

    var quiz = m.preguntas > 0 ? {
      requerida: true,
      preguntas: m.preguntas,
      aprobada: !!(res.quiz && res.quiz.aprobado),
      mejorPuntaje: res.quiz ? res.quiz.mejor : null,
      intentos: res.quiz ? res.quiz.intentos : 0
    } : null;
    var practica = m.tipo === 'practica' ? {
      requerida: true,
      aprobada: !!(res.practica && res.practica.aprobado),
      registrada: !!res.practica,
      puntaje: res.practica ? res.practica.puntaje : null,
      fecha: res.practica ? res.practica.fecha : '',
      evaluador: res.practica ? res.practica.evaluador : '',
      comentario: res.practica ? res.practica.comentario : ''
    } : null;

    var total = obligatorias.length + (quiz ? 1 : 0) + (practica ? 1 : 0);
    var hechasModulo = hechas + (quiz && quiz.aprobada ? 1 : 0) + (practica && practica.aprobada ? 1 : 0);
    var completo = total > 0 ? hechasModulo === total : true;
    var bloqueado = ordenObligatorio && !anteriorCompleto;

    unidades += total;
    unidadesHechas += hechasModulo;

    if (!siguiente && !bloqueado && !completo) {
      var pendiente = lecciones.filter(function (l) { return l.obligatoria && !l.completada; })[0];
      if (pendiente) siguiente = { moduloId: m.id, leccionId: pendiente.id };
      else if (quiz && !quiz.aprobada) siguiente = { moduloId: m.id, evaluacion: true };
      else siguiente = { moduloId: m.id };
    }
    anteriorCompleto = anteriorCompleto && completo;

    return Object.assign({}, m, {
      lecciones: lecciones,
      completadas: hechas,
      obligatorias: obligatorias.length,
      leccionesCompletas: leccionesCompletas,
      evaluacion: quiz,
      practica: practica,
      progreso: total ? Math.round(hechasModulo / total * 100) : 100,
      completo: completo,
      bloqueado: bloqueado,
      evaluacionDisponible: !!quiz && !bloqueado && leccionesCompletas
    });
  });

  var modulosCompletos = modulos.filter(function (m) { return m.completo; }).length;
  return {
    modulos: modulos,
    avance: unidades ? Math.round(unidadesHechas / unidades * 100) : 0,
    leccionesCompletadas: leccionesHechas,
    modulosCompletos: modulosCompletos,
    totalModulos: modulos.length,
    terminado: modulos.length > 0 && modulosCompletos === modulos.length,
    siguiente: siguiente
  };
}

function estadoUsuario_(email) {
  var estructura = leerEstructura_();
  return calcularEstado_(
    estructura,
    leerProgresoUsuario_(email).completadas,
    leerResultadosUsuario_(email),
    { ordenObligatorio: configSi_('ORDEN_OBLIGATORIO') }
  );
}

/** Marca o desmarca una lección. Devuelve el módulo al que pertenece. */
function registrarLeccion_(email, idLeccion, completada) {
  var estado = estadoUsuario_(email);
  var modulo = null;
  var leccion = null;
  estado.modulos.forEach(function (m) {
    m.lecciones.forEach(function (l) { if (l.id === idLeccion) { modulo = m; leccion = l; } });
  });
  if (!leccion) throw new Error('La lección no existe o ya no está visible.');
  if (modulo.bloqueado) throw new Error('Este módulo se desbloquea al completar el anterior.');

  conBloqueo_(function () {
    var progreso = leerProgresoUsuario_(email);
    var fila = progreso.filas[idLeccion];
    var valores = { 'Email': email, 'ID lección': idLeccion, 'ID módulo': modulo.id, 'Completada': completada ? 'SI' : 'NO', 'Fecha': new Date() };
    if (fila) actualizarFila_(HOJAS.PROGRESO, fila, valores);
    else anexarFilas_(HOJAS.PROGRESO, [valores]);
  });
  return modulo.id;
}

/* ------------------------------------------------------------------------------------------ */
/* Perfil y resumen en la hoja «Usuarios»                                                     */
/* ------------------------------------------------------------------------------------------ */

function perfilUsuario_(email) {
  var fila = leerTabla_(HOJAS.USUARIOS).filter(function (u) { return aTexto_(u['Email']).toLowerCase() === email; })[0];
  if (!fila) return null;
  return {
    fila: fila._fila,
    nombre: aTexto_(fila['Nombre']),
    area: aTexto_(fila['Área']),
    primerIngreso: aFechaIso_(fila['Primer ingreso']),
    ultimoIngreso: aFechaIso_(fila['Último ingreso']),
    certificado: aFechaIso_(fila['Certificado'])
  };
}

/** Registra la visita (crea la fila del colaborador la primera vez). */
function registrarIngreso_(email) {
  return conBloqueo_(function () {
    var perfil = perfilUsuario_(email);
    var ahora = new Date();
    if (perfil) {
      actualizarFila_(HOJAS.USUARIOS, perfil.fila, { 'Último ingreso': ahora });
      return perfil;
    }
    anexarFilas_(HOJAS.USUARIOS, [{
      'Email': email, 'Nombre': '', 'Área': '', 'Primer ingreso': ahora, 'Último ingreso': ahora,
      'Lecciones completadas': 0, 'Avance %': 0, 'Módulos completados': 0, 'Certificado': ''
    }]);
    return perfilUsuario_(email);
  });
}

function guardarPerfil_(email, nombre, area) {
  var limpio = function (t, max) { return aTexto_(t).replace(/[<>]/g, '').replace(/\s+/g, ' ').slice(0, max); };
  var datos = { 'Nombre': limpio(nombre, 80), 'Área': limpio(area, 60) };
  if (!datos['Nombre']) throw new Error('Escribe tu nombre para continuar.');
  conBloqueo_(function () {
    var perfil = perfilUsuario_(email) || registrarIngreso_(email);
    actualizarFila_(HOJAS.USUARIOS, perfil.fila, datos);
  });
}

/** Actualiza las columnas de resumen y la fecha del certificado cuando corresponde. */
function actualizarResumen_(email, estado) {
  conBloqueo_(function () {
    var perfil = perfilUsuario_(email);
    if (!perfil) return;
    var cambios = {
      'Lecciones completadas': estado.leccionesCompletadas,
      'Avance %': estado.avance,
      'Módulos completados': estado.modulosCompletos + ' de ' + estado.totalModulos
    };
    if (estado.terminado && !perfil.certificado) cambios['Certificado'] = new Date();
    actualizarFila_(HOJAS.USUARIOS, perfil.fila, cambios);
  });
}
