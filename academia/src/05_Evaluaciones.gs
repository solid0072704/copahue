/**
 * TPDV Academy · Evaluaciones.
 *  - Cuestionario del módulo: selección múltiple, se corrige en el servidor (las respuestas
 *    correctas nunca viajan al navegador antes de enviar).
 *  - Evaluación práctica (role play, casos): la registra un administrador desde el panel.
 */

var LETRAS = ['A', 'B', 'C', 'D'];

function preguntasDeModulo_(idModulo) {
  return leerTabla_(HOJAS.PREGUNTAS)
    .filter(function (p) { return aTexto_(p['ID módulo']) === idModulo && aTexto_(p['Pregunta']); })
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

function validarIntentos_(modulo) {
  var maximo = configNumero_('INTENTOS_MAXIMOS', 0, 50);
  if (maximo > 0 && modulo.evaluacion && !modulo.evaluacion.aprobada && modulo.evaluacion.intentos >= maximo) {
    throw new Error('Usaste los ' + maximo + ' intentos de esta evaluación. Conversa con tu tutora o supervisor para habilitar uno nuevo.');
  }
}

/** Preguntas para mostrar al colaborador (sin la respuesta correcta). */
function evaluacionParaAlumno_(email, idModulo) {
  var modulo = moduloDelUsuario_(email, idModulo);
  if (!modulo.evaluacion) throw new Error('Este módulo no tiene evaluación.');
  if (modulo.bloqueado) throw new Error('Este módulo se desbloquea al completar el anterior.');
  if (!modulo.leccionesCompletas) throw new Error('Completa todas las lecciones del módulo antes de rendir la evaluación.');
  validarIntentos_(modulo);
  return {
    moduloId: modulo.id,
    titulo: modulo.titulo,
    puntajeAprobacion: configNumero_('PUNTAJE_APROBACION', 0, 100),
    intentos: modulo.evaluacion.intentos,
    intentosMaximos: configNumero_('INTENTOS_MAXIMOS', 0, 50),
    preguntas: preguntasDeModulo_(idModulo).map(function (p) {
      return { n: p.n, pregunta: p.pregunta, opciones: p.opciones };
    })
  };
}

function registrarIntento_(email, idModulo, respuestas) {
  var modulo = moduloDelUsuario_(email, idModulo);
  if (!modulo.evaluacionDisponible) throw new Error('La evaluación de este módulo aún no está disponible.');
  validarIntentos_(modulo);
  var preguntas = preguntasDeModulo_(idModulo);
  if (!preguntas.length) throw new Error('Este módulo no tiene preguntas cargadas.');
  var resultado = corregir_(preguntas, respuestas, configNumero_('PUNTAJE_APROBACION', 0, 100));

  conBloqueo_(function () {
    anexarFilas_(HOJAS.RESULTADOS, [{
      'Fecha': new Date(),
      'Email': email,
      'ID módulo': idModulo,
      'Tipo': 'Cuestionario',
      'Puntaje %': resultado.puntaje,
      'Aprobado': resultado.aprobado ? 'SI' : 'NO',
      'Intento': modulo.evaluacion.intentos + 1,
      'Evaluador': 'Automático',
      'Comentario': resultado.detalle.map(function (d) { return d.n + ':' + (d.elegida || '-'); }).join(' ')
    }]);
  });

  // La respuesta correcta se muestra solo cuando se aprueba; antes, solo si acertó y la explicación.
  resultado.detalle.forEach(function (d) {
    if (!resultado.aprobado) delete d.respuestaCorrecta;
  });
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
