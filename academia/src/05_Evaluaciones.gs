/**
 * TPDV Academy · Evaluaciones.
 *  - Mini evaluación de cada video (preguntas con «ID lección»): al aprobarla se completa la lección.
 *  - Evaluación final de cada módulo (preguntas sin «ID lección»): se habilita al completar las lecciones.
 *  - Evaluación práctica (role play, casos): la registra un administrador desde el panel; no cuenta en la barra
 *    de avance, pero puede ser requisito del certificado (CERTIFICADO_REQUIERE_PRACTICA).
 * Todas se corrigen en el servidor: las respuestas correctas nunca viajan al navegador antes de enviar.
 */

var LETRAS = ['A', 'B', 'C', 'D'];

/** Preguntas de la evaluación final del módulo (idLeccion vacío) o de la mini evaluación de una lección. */
function preguntasDeModulo_(idModulo, idLeccion) {
  var leccion = aTexto_(idLeccion);
  return leerTabla_(HOJAS.PREGUNTAS)
    .filter(function (p) {
      return aTexto_(p['ID módulo']) === idModulo && aTexto_(p['ID lección']) === leccion && aTexto_(p['Pregunta']);
    })
    .map(function (p, i) {
      return {
        n: Number(p['N°']) || i + 1,
        pregunta: aTexto_(p['Pregunta']),
        opciones: LETRAS.filter(function (l) { return aTexto_(p[l]); }).map(function (l) {
          return { letra: l, texto: aTexto_(p[l]) };
        }),
        correcta: aTexto_(p['Correcta']).toUpperCase().charAt(0),
        explicacion: aTexto_(p['Explicación'])
      };
    })
    .sort(function (a, b) { return a.n - b.n; });
}

/** Corrige las respuestas {n: letra}. Función pura. */
function corregir_(preguntas, respuestas, puntajeAprobacion) {
  respuestas = respuestas || {};
  var correctas = 0;
  var detalle = preguntas.map(function (p) {
    var elegida = aTexto_(respuestas[p.n]).toUpperCase().charAt(0);
    var ok = elegida !== '' && elegida === p.correcta;
    if (ok) correctas++;
    return { n: p.n, elegida: elegida, correcta: ok, explicacion: p.explicacion, respuestaCorrecta: p.correcta };
  });
  var total = preguntas.length;
  var puntaje = total ? Math.round(correctas / total * 100) : 0;
  return { correctas: correctas, total: total, puntaje: puntaje, aprobado: puntaje >= puntajeAprobacion, detalle: detalle };
}

function moduloDelUsuario_(email, idModulo) {
  var estado = estadoUsuario_(email);
  var modulo = estado.modulos.filter(function (m) { return m.id === idModulo; })[0];
  if (!modulo) throw new Error('El módulo no existe o ya no está visible.');
  return modulo;
}

function leccionDelModulo_(modulo, idLeccion) {
  var leccion = modulo.lecciones.filter(function (l) { return l.id === idLeccion; })[0];
  if (!leccion) throw new Error('La lección no existe o ya no está visible.');
  if (!leccion.miniEvaluacion) throw new Error('Esta lección no tiene mini evaluación.');
  return leccion;
}

function validarIntentos_(modulo) {
  var maximo = configNumero_('INTENTOS_MAXIMOS', 0, 50);
  if (maximo > 0 && modulo.evaluacion && !modulo.evaluacion.aprobada && modulo.evaluacion.intentos >= maximo) {
    throw new Error('Usaste los ' + maximo + ' intentos de esta evaluación. Conversa con tu tutora o supervisor para habilitar uno nuevo.');
  }
}

/** Preguntas para mostrar al colaborador (sin la respuesta correcta). */
function evaluacionParaAlumno_(email, idModulo, idLeccion) {
  var modulo = moduloDelUsuario_(email, idModulo);
  if (modulo.bloqueado) throw new Error('Este módulo se desbloquea al completar el anterior.');
  var esMini = !!aTexto_(idLeccion);
  var base;
  if (esMini) {
    var leccion = leccionDelModulo_(modulo, idLeccion);
    base = {
      tipo: 'mini', leccionId: leccion.id, titulo: leccion.titulo,
      puntajeAprobacion: configNumero_('PUNTAJE_APROBACION_MINI', 0, 100),
      intentos: leccion.miniEvaluacion.intentos, intentosMaximos: 0
    };
  } else {
    if (!modulo.evaluacion) throw new Error('Este módulo no tiene evaluación final.');
    if (!modulo.leccionesCompletas) throw new Error('Completa todas las lecciones del módulo (y sus mini evaluaciones) antes de rendir la evaluación final.');
    validarIntentos_(modulo);
    base = {
      tipo: 'final', leccionId: '', titulo: modulo.titulo,
      puntajeAprobacion: configNumero_('PUNTAJE_APROBACION', 0, 100),
      intentos: modulo.evaluacion.intentos, intentosMaximos: configNumero_('INTENTOS_MAXIMOS', 0, 50)
    };
  }
  base.moduloId = modulo.id;
  base.preguntas = preguntasDeModulo_(idModulo, esMini ? idLeccion : '').map(function (p) {
    return { n: p.n, pregunta: p.pregunta, opciones: p.opciones };
  });
  return base;
}

function registrarIntento_(email, idModulo, respuestas, idLeccion) {
  var modulo = moduloDelUsuario_(email, idModulo);
  var esMini = !!aTexto_(idLeccion);
  var intento;
  var umbral;
  if (esMini) {
    if (modulo.bloqueado) throw new Error('Este módulo se desbloquea al completar el anterior.');
    intento = leccionDelModulo_(modulo, idLeccion).miniEvaluacion.intentos + 1;
    umbral = configNumero_('PUNTAJE_APROBACION_MINI', 0, 100);
  } else {
    if (!modulo.evaluacionDisponible) throw new Error('La evaluación final de este módulo aún no está disponible.');
    validarIntentos_(modulo);
    intento = modulo.evaluacion.intentos + 1;
    umbral = configNumero_('PUNTAJE_APROBACION', 0, 100);
  }
  var preguntas = preguntasDeModulo_(idModulo, esMini ? idLeccion : '');
  if (!preguntas.length) throw new Error('Esta evaluación no tiene preguntas cargadas.');
  var resultado = corregir_(preguntas, respuestas, umbral);

  conBloqueo_(function () {
    anexarFilas_(HOJAS.RESULTADOS, [{
      'Fecha': new Date(),
      'Email': email,
      'ID módulo': idModulo,
      'ID lección': esMini ? idLeccion : '',
      'Tipo': esMini ? 'Mini evaluación' : 'Evaluación final',
      'Puntaje %': resultado.puntaje,
      'Aprobado': resultado.aprobado ? 'SI' : 'NO',
      'Intento': intento,
      'Evaluador': 'Automático',
      'Comentario': resultado.detalle.map(function (d) { return d.n + ':' + (d.elegida || '-'); }).join(' ')
    }]);
  });
  // La mini evaluación aprobada completa la lección (queda la fecha en «Progreso»).
  if (esMini && resultado.aprobado) registrarLeccion_(email, idLeccion, true, true);

  // La respuesta correcta se muestra solo cuando se aprueba; antes, solo si acertó y la explicación.
  resultado.detalle.forEach(function (d) {
    if (!resultado.aprobado) delete d.respuestaCorrecta;
  });
  resultado.tipo = esMini ? 'mini' : 'final';
  resultado.umbral = umbral;
  return resultado;
}

/** Registro de una evaluación práctica (role play, casos) hecho por un administrador. */
function registrarPractica_(evaluador, emailColaborador, idModulo, puntaje, comentario) {
  var email = aTexto_(emailColaborador).toLowerCase();
  if (!esCorreoPermitido_(email)) throw new Error('El correo del colaborador debe ser @' + DOMINIO_PERMITIDO + '.');
  var modulo = leerEstructura_().modulos.filter(function (m) { return m.id === idModulo; })[0];
  if (!modulo || modulo.tipo !== 'practica') throw new Error('El módulo indicado no tiene evaluación práctica.');
  var nota = Math.round(Number(puntaje));
  if (!isFinite(nota) || nota < 0 || nota > 100) throw new Error('El puntaje debe ser un número entre 0 y 100.');
  var aprobado = nota >= configNumero_('PUNTAJE_APROBACION', 0, 100);
  conBloqueo_(function () {
    anexarFilas_(HOJAS.RESULTADOS, [{
      'Fecha': new Date(),
      'Email': email,
      'ID módulo': idModulo,
      'ID lección': '',
      'Tipo': 'Práctica',
      'Puntaje %': nota,
      'Aprobado': aprobado ? 'SI' : 'NO',
      'Intento': '',
      'Evaluador': evaluador,
      'Comentario': aTexto_(comentario).replace(/[<>]/g, '').slice(0, 1000)
    }]);
  });
  actualizarResumen_(email, estadoUsuario_(email));
  return { aprobado: aprobado, puntaje: nota };
}
