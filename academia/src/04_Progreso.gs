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
 * Resultados por módulo:
 * { idModulo: { quiz: {...}, mini: { idLeccion: {...} }, practica: {...} } }
 * quiz y mini: { mejor, aprobado, intentos, ultima }; practica: { puntaje, aprobado, fecha, evaluador, comentario }.
 */
function leerResultadosUsuario_(email, filasPrecargadas) {
  var mapa = {};
  var acumular = function (actual, puntaje, aprobado, fecha) {
    var q = actual || { mejor: 0, aprobado: false, intentos: 0, ultima: '' };
    q.intentos += 1;
    q.mejor = Math.max(q.mejor, puntaje);
    q.aprobado = q.aprobado || aprobado;
    if (fecha > q.ultima) q.ultima = fecha;
    return q;
  };
  (filasPrecargadas || leerTabla_(HOJAS.RESULTADOS)).forEach(function (r) {
    if (aTexto_(r['Email']).toLowerCase() !== email) return;
    var id = aTexto_(r['ID módulo']);
    var idLeccion = aTexto_(r['ID lección']);
    var tipo = aTexto_(r['Tipo']).toLowerCase();
    mapa[id] = mapa[id] || { mini: {} };
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
    } else if (idLeccion) {
      mapa[id].mini[idLeccion] = acumular(mapa[id].mini[idLeccion], puntaje, aprobado, fecha);
    } else {
      mapa[id].quiz = acumular(mapa[id].quiz, puntaje, aprobado, fecha);
    }
  });
  return mapa;
}

/**
 * Calcula el estado del programa para un usuario (función pura).
 *  - Avance (barra): lecciones obligatorias + evaluación final de cada módulo. Una lección con mini evaluación
 *    se completa al aprobarla. La evaluación práctica de la tutora NO cuenta en el avance: al terminar los
 *    cursos la barra llega a 100%.
 *  - Certificado: todos los módulos completos y, si así se configura, las evaluaciones prácticas aprobadas.
 * opciones: { ordenObligatorio: bool, certificadoRequierePractica: bool }
 */
function calcularEstado_(estructura, completadas, resultados, opciones) {
  opciones = opciones || {};
  var ordenObligatorio = opciones.ordenObligatorio !== false;
  var requierePractica = opciones.certificadoRequierePractica !== false;
  var unidades = 0;
  var unidadesHechas = 0;
  var leccionesHechas = 0;
  var anteriorCompleto = true;
  var siguiente = null;

  var modulos = estructura.modulos.map(function (m) {
    var res = resultados[m.id] || {};
    var minis = res.mini || {};
    var lecciones = m.lecciones.map(function (l) {
      var r = minis[l.id];
      var mini = l.preguntas > 0 ? {
        preguntas: l.preguntas,
        aprobada: !!(r && r.aprobado),
        mejorPuntaje: r ? r.mejor : null,
        intentos: r ? r.intentos : 0
      } : null;
      var hecha = mini ? mini.aprobada : !!completadas[l.id];
      if (hecha) leccionesHechas++;
      return Object.assign({}, l, { completada: hecha, miniEvaluacion: mini });
    });
    var obligatorias = lecciones.filter(function (l) { return l.obligatoria; });
    var hechas = obligatorias.filter(function (l) { return l.completada; }).length;
    var leccionesCompletas = hechas === obligatorias.length;

    var quiz = m.preguntas > 0 ? {
      requerida: true,
      preguntas: m.preguntas,
      aprobada: !!(res.quiz && res.quiz.aprobado),
      mejorPuntaje: res.quiz ? res.quiz.mejor : null,
      intentos: res.quiz ? res.quiz.intentos : 0
    } : null;
    var practica = m.tipo === 'practica' ? {
      requerida: requierePractica,
      aprobada: !!(res.practica && res.practica.aprobado),
      registrada: !!res.practica,
      puntaje: res.practica ? res.practica.puntaje : null,
      fecha: res.practica ? res.practica.fecha : '',
      evaluador: res.practica ? res.practica.evaluador : '',
      comentario: res.practica ? res.practica.comentario : ''
    } : null;

    var total = obligatorias.length + (quiz ? 1 : 0);
    var hechasModulo = hechas + (quiz && quiz.aprobada ? 1 : 0);
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
      practicaPendiente: !!practica && !practica.aprobada,
      progreso: total ? Math.round(hechasModulo / total * 100) : 100,
      completo: completo,
      bloqueado: bloqueado,
      evaluacionDisponible: !!quiz && !bloqueado && leccionesCompletas
    });
  });

  var modulosCompletos = modulos.filter(function (m) { return m.completo; }).length;
  var terminado = modulos.length > 0 && modulosCompletos === modulos.length;
  var practicasPendientes = modulos.filter(function (m) { return m.practicaPendiente; }).length;
  return {
    modulos: modulos,
    avance: unidades ? Math.round(unidadesHechas / unidades * 100) : 0,
    leccionesCompletadas: leccionesHechas,
    modulosCompletos: modulosCompletos,
    totalModulos: modulos.length,
    terminado: terminado,
    certificadoDisponible: terminado && (!requierePractica || practicasPendientes === 0),
    esperaPractica: terminado && requierePractica && practicasPendientes > 0,
    siguiente: siguiente
  };
}

function estadoUsuario_(email) {
  var estructura = leerEstructura_();
  return calcularEstado_(
    estructura,
    leerProgresoUsuario_(email).completadas,
    leerResultadosUsuario_(email),
    { ordenObligatorio: configSi_('ORDEN_OBLIGATORIO'), certificadoRequierePractica: configSi_('CERTIFICADO_REQUIERE_PRACTICA') }
  );
}

/**
 * Marca o desmarca una lección. Las lecciones con mini evaluación solo se completan al aprobarla
 * (desdeEvaluacion = true lo usa el registro de la mini evaluación).
 */
function registrarLeccion_(email, idLeccion, completada, desdeEvaluacion) {
  var estado = estadoUsuario_(email);
  var modulo = null;
  var leccion = null;
  estado.modulos.forEach(function (m) {
    m.lecciones.forEach(function (l) { if (l.id === idLeccion) { modulo = m; leccion = l; } });
  });
  if (!leccion) throw new Error('La lección no existe o ya no está visible.');
  if (modulo.bloqueado) throw new Error('Este módulo se desbloquea al completar el anterior.');
  if (leccion.miniEvaluacion && !desdeEvaluacion) {
    throw new Error('Esta lección se completa al aprobar su mini evaluación.');
  }

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
    if (estado.certificadoDisponible && !perfil.certificado) cambios['Certificado'] = new Date();
    actualizarFila_(HOJAS.USUARIOS, perfil.fila, cambios);
  });
}
