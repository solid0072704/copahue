/**
 * Pruebas del servidor de TPDV Academy con el simulador de Apps Script.
 * Ejecutar:  node --test academia/tests
 */
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { crearEntorno } = require('./gas_mock');

const { ALUMNA, ADMIN, COORDINACION, entornoInstalado, respuestasCorrectas, completarModulo } = require('./fixtures');

/* ------------------------------------------------------------------ Acceso */

test('solo se aceptan correos exactos @tuproyectodevida.cl', () => {
  const { contexto } = crearEntorno();
  const permitidos = ['ana@tuproyectodevida.cl', 'ANA.PEREZ@TuProyectoDeVida.CL', 'a+b@tuproyectodevida.cl'];
  const rechazados = ['', 'ana@gmail.com', 'ana@tuproyectodevida.cl.evil.com', 'ana@sub.tuproyectodevida.cl',
    'ana@tuproyectodevida.com', 'tuproyectodevida.cl', '@tuproyectodevida.cl', 'ana@tuproyectodevida.cl@gmail.com',
    'ana tuproyectodevida.cl', 'ana@xtuproyectodevida.cl', null, undefined];
  permitidos.forEach(e => assert.equal(contexto.esCorreoPermitido_(e), true, e));
  rechazados.forEach(e => assert.equal(contexto.esCorreoPermitido_(e), false, String(e)));
});

test('una cuenta externa ve la página de acceso restringido y no recibe datos', () => {
  const env = entornoInstalado();
  env.comoUsuario('alguien@gmail.com');
  const html = env.ejecutar('doGet').getContent();
  assert.match(html, /es exclusiva para el equipo/);
  assert.match(html, /alguien@gmail\.com/);
  assert.doesNotMatch(html, /DATOS_INICIALES/);
  assert.doesNotMatch(html, /VENTAS-1/);
});

test('sin correo (cuenta no identificada) también se bloquea', () => {
  const env = entornoInstalado();
  env.comoUsuario('');
  assert.match(env.ejecutar('doGet').getContent(), /No pudimos identificar tu cuenta/);
});

test('todas las funciones públicas rechazan cuentas externas', () => {
  const env = entornoInstalado();
  env.comoUsuario('externo@otraempresa.cl');
  const llamadas = [
    ['obtenerDatos'], ['marcarLeccion', 'IND-01', true], ['guardarPerfil', { nombre: 'X' }],
    ['obtenerEvaluacion', 'INDUCCION'], ['enviarEvaluacion', 'INDUCCION', {}], ['obtenerPanelAdmin'],
    ['sincronizarContenido'], ['verificarPermisos'], ['registrarEvaluacionPractica', ALUMNA, 'FINAL', 90, ''],
    ['obtenerDetalleColaborador', ALUMNA]
  ];
  llamadas.forEach(([fn, ...args]) => assert.throws(() => env.ejecutar(fn, ...args), /tuproyectodevida\.cl|administradores/, fn));
  // Funciones de mantenimiento llamadas desde el navegador por alguien externo.
  ['instalar', 'activarSincronizacionDiaria', 'sincronizacionProgramada'].forEach(fn =>
    assert.throws(() => env.ejecutar(fn), /administradores/, fn));
});

test('un colaborador del dominio que no es administrador no accede al panel', () => {
  const env = entornoInstalado();
  env.comoUsuario(ALUMNA);
  assert.equal(env.ejecutar('obtenerDatos').usuario.admin, false);
  assert.throws(() => env.ejecutar('obtenerPanelAdmin'), /administradores/);
  assert.throws(() => env.ejecutar('registrarEvaluacionPractica', ALUMNA, 'FINAL', 100, ''), /administradores/);
  assert.throws(() => env.ejecutar('instalar'), /administradores/);
});

test('solo las cuentas de la lista de administradores ven el panel', () => {
  const env = entornoInstalado();
  env.comoUsuario(COORDINACION.toUpperCase());
  assert.equal(env.ejecutar('obtenerDatos').usuario.admin, true, 'sin importar mayúsculas');
  assert.ok(env.ejecutar('obtenerPanelAdmin').resumen);

  // La planilla no puede agregar administradores.
  env.hoja('Config').datos.push(['ADMINISTRADORES', 'jefa.ventas@tuproyectodevida.cl', '']);
  env.comoUsuario('jefa.ventas@tuproyectodevida.cl');
  assert.equal(env.ejecutar('obtenerDatos').usuario.admin, false);
  assert.throws(() => env.ejecutar('obtenerPanelAdmin'), /administradores/);
  assert.ok(!env.contexto.CONFIG_DEFECTO.some(f => f[0] === 'ADMINISTRADORES'), 'no existe la clave en la planilla');

  // El dueño del script no es administrador si no está en la lista, y los correos externos se ignoran.
  const soloUno = crearEntorno({ dueno: 'dueno.script@tuproyectodevida.cl', usuario: 'dueno.script@tuproyectodevida.cl',
    administradores: [COORDINACION, 'intruso@gmail.com'] });
  soloUno.ejecutar('instalar');
  assert.equal(soloUno.ejecutar('obtenerDatos').usuario.admin, false);
  assert.deepEqual(Array.from(soloUno.contexto.listaAdministradores_()), [COORDINACION]);
  assert.equal(soloUno.contexto.esAdmin_('intruso@gmail.com'), false);
});

test('sin lista de administradores nadie ve el panel', () => {
  const env = crearEntorno({ dueno: ADMIN, usuario: ADMIN, administradores: [] });
  const texto = env.ejecutar('instalar');
  assert.match(texto, /Nadie puede ver el panel/);
  assert.equal(env.ejecutar('obtenerDatos').usuario.admin, false);
  assert.throws(() => env.ejecutar('obtenerPanelAdmin'), /administradores/);
});

test('el panel muestra el avance de cada colaborador por módulo y el detalle individual', () => {
  const env = entornoInstalado();
  env.comoUsuario(COORDINACION);
  env.ejecutar('doGet'); // el administrador también queda registrado, pero no cuenta en los indicadores
  env.comoUsuario(ALUMNA);
  env.ejecutar('doGet');
  env.ejecutar('guardarPerfil', { nombre: 'Camila Rojas', area: 'Ventas' });
  completarModulo(env, 'INDUCCION');
  env.ejecutar('enviarEvaluacion', 'VENTAS-1', respuestasCorrectas(env, 'VENTAS-1', 'V1-01'), 'V1-01');
  env.ejecutar('marcarLeccion', 'V1-02', true);
  env.ejecutar('enviarEvaluacion', 'VENTAS-1', { 1: 'A', 2: 'A', 3: 'A', 4: 'A' }, 'V1-09');
  env.comoUsuario('otro.asesor@tuproyectodevida.cl');
  env.ejecutar('doGet');

  env.comoUsuario(COORDINACION);
  const panel = env.ejecutar('obtenerPanelAdmin');
  assert.equal(panel.resumen.colaboradores, 2, 'los administradores no cuentan');
  const camila = panel.colaboradores.find(c => c.email === ALUMNA);
  assert.deepEqual(camila.modulos.map(m => m.estado), ['completo', 'curso', 'bloqueado', 'bloqueado', 'bloqueado', 'bloqueado', 'bloqueado', 'bloqueado']);
  assert.equal(camila.modulos[0].progreso, 100);
  assert.equal(camila.modulos[1].progreso, 14);
  assert.equal(camila.diasSinIngreso, 0);
  assert.equal(panel.colaboradores.find(c => c.email === COORDINACION).esAdmin, true);
  assert.equal(panel.colaboradores.find(c => c.email === 'otro.asesor@tuproyectodevida.cl').modulos[0].estado, 'pendiente');
  assert.deepEqual(panel.modulos.map(m => m.corto), ['Inducción', 'Ventas I', 'Ventas II', 'Marketing I', 'Marketing II', 'Legal I', 'Legal II', 'Evaluación final']);

  const d = env.ejecutar('obtenerDetalleColaborador', ALUMNA.toUpperCase());
  assert.equal(d.perfil.nombre, 'Camila Rojas');
  assert.equal(d.leccionesCompletadas, 5);
  assert.equal(d.totalLecciones, env.catalogo.lecciones.length);
  const v1 = d.modulos.find(m => m.id === 'VENTAS-1');
  assert.equal(v1.estado, 'curso');
  assert.ok(v1.lecciones.find(l => l.id === 'V1-01').fecha, 'fecha de la lección completada');
  assert.equal(v1.lecciones.find(l => l.id === 'V1-03').fecha, '');
  assert.equal(v1.lecciones.find(l => l.id === 'V1-01').miniEvaluacion.aprobada, true);
  assert.equal(v1.lecciones.find(l => l.id === 'V1-09').miniEvaluacion.intentos, 1);
  assert.equal(v1.lecciones.find(l => l.id === 'V1-09').completada, false);
  assert.deepEqual(d.historial.map(h => h.tipo).sort(), ['Evaluación final', 'Mini evaluación', 'Mini evaluación']);
  assert.ok(d.historial.some(h => h.tipo === 'Mini evaluación' && /HubSpot 1/.test(h.modulo)), 'el historial nombra el video');
  assert.equal(d.historial.find(h => h.tipo === 'Evaluación final').aprobado, true);
  assert.ok(panel.modulos.find(m => m.id === 'VENTAS-1').minis >= 3, 'el panel cuenta las mini evaluaciones');
  assert.ok(d.perfil.ultimaActividad);

  assert.throws(() => env.ejecutar('obtenerDetalleColaborador', 'nadie@tuproyectodevida.cl'), /aún no ingresa/);
  assert.throws(() => env.ejecutar('obtenerDetalleColaborador', 'alguien@gmail.com'), /debe ser @/);
  env.comoUsuario(ALUMNA);
  assert.throws(() => env.ejecutar('obtenerDetalleColaborador', ALUMNA), /administradores/);
});

/* ------------------------------------------------------------------ Instalación */

test('instalar crea la base de datos con el programa y es idempotente', () => {
  const env = entornoInstalado();
  const nombres = env.libro().getSheets().map(h => h.getName());
  ['Config', 'Módulos', 'Lecciones', 'Preguntas', 'Usuarios', 'Progreso', 'Resultados', 'Por clasificar'].forEach(n => assert.ok(nombres.includes(n), n));
  assert.ok(!nombres.includes('Hoja 1'));
  const conteo = () => ['Módulos', 'Lecciones', 'Preguntas', 'Config'].map(n => env.hoja(n).objetos().length);
  const antes = conteo();
  assert.deepEqual(antes.slice(0, 3), [env.catalogo.modulos.length, env.catalogo.lecciones.length, env.catalogo.preguntas.length]);
  env.ejecutar('instalar');
  assert.deepEqual(conteo(), antes, 'la segunda instalación no duplica filas');
});

test('instalar avisa si la cuenta dueña no es del dominio', () => {
  const env = crearEntorno({ dueno: 'persona@gmail.com', usuario: 'persona@gmail.com' });
  const texto = env.ejecutar('instalar');
  assert.match(texto, /no es @tuproyectodevida\.cl/);
});

/* ------------------------------------------------------------------ Recorrido del colaborador */

test('recorrido completo: lecciones, evaluaciones, bloqueo secuencial y certificado', () => {
  const env = entornoInstalado();
  env.comoUsuario(ALUMNA);

  const pagina = env.ejecutar('doGet').getContent();
  assert.match(pagina, /window\.DATOS_INICIALES = \{/);
  assert.match(pagina, /--primario: #005A82/);
  assert.match(pagina, /family=Montserrat/);
  assert.match(pagina, /family=Lato/);
  assert.match(pagina, /src="data:image\/png;base64,/, 'el logo se incrusta como data URI');

  let datos = env.ejecutar('obtenerDatos');
  assert.equal(datos.programa.avance, 0);
  assert.equal(datos.usuario.nombre, '');
  const [ind, v1] = datos.programa.modulos;
  assert.equal(ind.id, 'INDUCCION');
  assert.equal(ind.bloqueado, false);
  assert.equal(v1.bloqueado, true);
  assert.deepEqual(datos.programa.siguiente, { moduloId: 'INDUCCION', leccionId: 'IND-01' });
  assert.match(datos.contenidos['IND-01'], /80%/, 'el puntaje se inserta en la lectura');
  assert.throws(() => env.ejecutar('marcarLeccion', v1.lecciones[0].id, true), /se desbloquea/);
  assert.throws(() => env.ejecutar('obtenerEvaluacion', 'INDUCCION'), /Completa todas las lecciones/);
  assert.throws(() => env.ejecutar('marcarLeccion', 'NO-EXISTE', true), /no existe/);

  datos = env.ejecutar('guardarPerfil', { nombre: '  Camila   Rojas <b> ', area: 'Ventas' });
  assert.equal(datos.usuario.nombre, 'Camila Rojas b');
  assert.throws(() => env.ejecutar('guardarPerfil', { nombre: '   ' }), /Escribe tu nombre/);

  ind.lecciones.forEach(l => { datos = env.ejecutar('marcarLeccion', l.id, true); });
  let modInd = datos.programa.modulos[0];
  assert.equal(modInd.leccionesCompletas, true);
  assert.equal(modInd.evaluacionDisponible, true);
  assert.equal(modInd.completo, false, 'falta aprobar la evaluación');

  const ev = env.ejecutar('obtenerEvaluacion', 'INDUCCION');
  assert.equal(ev.preguntas.length, 6);
  assert.ok(!JSON.stringify(ev).includes('"correcta"'), 'no se envía la respuesta correcta');
  assert.ok(!JSON.stringify(ev).includes('explicacion'));

  const malas = env.ejecutar('enviarEvaluacion', 'INDUCCION', { 1: 'A', 2: 'A', 3: 'A', 4: 'A', 5: 'A', 6: 'A' });
  assert.equal(malas.resultado.aprobado, false);
  assert.equal(malas.resultado.puntaje, 17);
  assert.ok(malas.resultado.detalle.every(d => d.respuestaCorrecta === undefined), 'sin aprobar no se revelan las correctas');

  const buenas = env.ejecutar('enviarEvaluacion', 'INDUCCION', respuestasCorrectas(env, 'INDUCCION'));
  assert.equal(buenas.resultado.aprobado, true);
  assert.equal(buenas.resultado.puntaje, 100);
  modInd = buenas.datos.programa.modulos[0];
  assert.equal(modInd.completo, true);
  assert.equal(modInd.evaluacion.intentos, 2);
  assert.equal(buenas.datos.programa.modulos[1].bloqueado, false, 'Ventas I se desbloquea');

  ['VENTAS-1', 'VENTAS-2', 'MARKETING-1', 'MARKETING-2', 'LEGAL-1', 'LEGAL-2'].forEach(id => completarModulo(env, id));
  datos = env.ejecutar('obtenerDatos');
  const final = datos.programa.modulos.find(m => m.id === 'FINAL');
  assert.equal(final.bloqueado, false);
  assert.ok(final.evaluacion && final.evaluacion.preguntas >= 5, 'el módulo final también tiene evaluación final');
  completarModulo(env, 'FINAL');
  datos = env.ejecutar('obtenerDatos');
  assert.equal(datos.programa.avance, 100, 'al terminar los cursos la barra llega a 100% aunque falte la práctica');
  assert.equal(datos.programa.terminado, true);
  assert.equal(datos.programa.modulos.find(m => m.id === 'FINAL').progreso, 100);
  assert.equal(datos.programa.modulos.find(m => m.id === 'FINAL').practicaPendiente, true);
  assert.equal(datos.certificado.disponible, false, 'el certificado espera la evaluación práctica');
  assert.equal(datos.certificado.esperaPractica, true);
  assert.equal(datos.programa.siguiente, null);
  assert.equal(env.hoja('Usuarios').objetos().find(u => u.Email === ALUMNA)['Avance %'], 100);

  // La tutora (administradora) registra la práctica: primero por reforzar, después aprobada.
  env.comoUsuario(ADMIN);
  let panel = env.ejecutar('obtenerPanelAdmin');
  const fila = panel.colaboradores.find(c => c.email === ALUMNA);
  assert.deepEqual(fila.practicasPendientes.map(p => p.id), ['FINAL']);
  assert.equal(fila.avance, 100);
  assert.equal(fila.modulos.find(m => m.id === 'FINAL').estado, 'practica');
  assert.equal(fila.certificado, '');
  assert.throws(() => env.ejecutar('registrarEvaluacionPractica', ALUMNA, 'FINAL', 140, ''), /entre 0 y 100/);
  assert.throws(() => env.ejecutar('registrarEvaluacionPractica', ALUMNA, 'VENTAS-1', 90, ''), /no tiene evaluación práctica/);
  assert.throws(() => env.ejecutar('registrarEvaluacionPractica', 'otra@gmail.com', 'FINAL', 90, ''), /debe ser @/);
  let r = env.ejecutar('registrarEvaluacionPractica', ALUMNA, 'FINAL', 60, 'Reforzar manejo de objeciones');
  assert.equal(r.resultado.aprobado, false);
  r = env.ejecutar('registrarEvaluacionPractica', ALUMNA, 'FINAL', 87, 'Muy buena presentación del proyecto');
  assert.equal(r.resultado.aprobado, true);
  panel = r.panel;
  assert.equal(panel.resumen.certificados, 1);
  assert.equal(panel.resumen.practicasPendientes, 0);

  env.comoUsuario(ALUMNA);
  datos = env.ejecutar('obtenerDatos');
  assert.equal(datos.programa.terminado, true);
  assert.equal(datos.programa.avance, 100);
  assert.equal(datos.certificado.disponible, true);
  assert.equal(datos.certificado.esperaPractica, false);
  assert.equal(datos.programa.modulos.find(m => m.id === 'FINAL').practica.comentario, 'Muy buena presentación del proyecto');
  const usuario = env.hoja('Usuarios').objetos().find(u => u.Email === ALUMNA);
  assert.equal(Object.prototype.toString.call(usuario.Certificado), '[object Date]', 'se registra la fecha del certificado');
  assert.equal(usuario['Avance %'], 100);
});

test('cada video tiene una mini evaluación de 4 preguntas que completa la lección', () => {
  const env = entornoInstalado();
  env.comoUsuario(ALUMNA);
  let datos = env.ejecutar('obtenerDatos');
  const videos = [];
  datos.programa.modulos.forEach(m => m.lecciones.forEach(l => { if (l.tipo === 'Video') videos.push({ m, l }); }));
  assert.ok(videos.length >= 8);
  videos.forEach(({ l }) => {
    assert.ok(l.miniEvaluacion, 'mini evaluación en ' + l.id);
    assert.equal(l.miniEvaluacion.preguntas, 4, '4 preguntas en ' + l.id);
  });
  datos.programa.modulos.forEach(m => m.lecciones.filter(l => l.tipo !== 'Video').forEach(l =>
    assert.equal(l.miniEvaluacion, null, 'solo los videos tienen mini evaluación: ' + l.id)));
  assert.equal(datos.academia.puntajeAprobacionMini, 75);

  // Módulo bloqueado: tampoco se rinde su mini evaluación.
  assert.throws(() => env.ejecutar('obtenerEvaluacion', 'VENTAS-1', 'V1-01'), /se desbloquea/);
  assert.throws(() => env.ejecutar('enviarEvaluacion', 'VENTAS-1', {}, 'V1-01'), /se desbloquea/);
  completarModulo(env, 'INDUCCION');

  // Un video no se marca a mano: se completa al aprobar su mini evaluación.
  assert.throws(() => env.ejecutar('marcarLeccion', 'V1-01', true), /mini evaluación/);
  assert.throws(() => env.ejecutar('obtenerEvaluacion', 'VENTAS-1', 'V1-02'), /no tiene mini evaluación/);
  const ev = env.ejecutar('obtenerEvaluacion', 'VENTAS-1', 'V1-01');
  assert.equal(ev.tipo, 'mini');
  assert.equal(ev.leccionId, 'V1-01');
  assert.equal(ev.preguntas.length, 4);
  assert.equal(ev.puntajeAprobacion, 75);
  assert.ok(!JSON.stringify(ev).includes('"correcta"'), 'no se envía la respuesta correcta');

  const correctas = respuestasCorrectas(env, 'VENTAS-1', 'V1-01');
  const otra = (letra) => (letra === 'A' ? 'B' : 'A');
  const dosBien = { 1: correctas[1], 2: correctas[2], 3: otra(correctas[3]), 4: otra(correctas[4]) };
  let r = env.ejecutar('enviarEvaluacion', 'VENTAS-1', dosBien, 'V1-01');
  assert.equal(r.resultado.tipo, 'mini');
  assert.equal(r.resultado.puntaje, 50);
  assert.equal(r.resultado.aprobado, false);
  assert.ok(r.resultado.detalle.every(d => d.respuestaCorrecta === undefined));
  let v1 = r.datos.programa.modulos.find(m => m.id === 'VENTAS-1');
  assert.equal(v1.lecciones.find(l => l.id === 'V1-01').completada, false);
  assert.equal(v1.lecciones.find(l => l.id === 'V1-01').miniEvaluacion.intentos, 1);

  const tresBien = Object.assign({}, dosBien, { 3: correctas[3] });
  r = env.ejecutar('enviarEvaluacion', 'VENTAS-1', tresBien, 'V1-01');
  assert.equal(r.resultado.puntaje, 75);
  assert.equal(r.resultado.aprobado, true, '3 de 4 aprueba');
  v1 = r.datos.programa.modulos.find(m => m.id === 'VENTAS-1');
  const leccion = v1.lecciones.find(l => l.id === 'V1-01');
  assert.equal(leccion.completada, true);
  assert.equal(leccion.miniEvaluacion.aprobada, true);
  assert.equal(leccion.miniEvaluacion.mejorPuntaje, 75);
  assert.ok(v1.progreso > 0);

  const progreso = env.hoja('Progreso').objetos().filter(f => f.Email === ALUMNA && f['ID lección'] === 'V1-01');
  assert.equal(progreso.length, 1, 'la mini aprobada deja la lección registrada en «Progreso»');
  const filas = env.hoja('Resultados').objetos().filter(f => f.Email === ALUMNA && f['ID lección'] === 'V1-01');
  assert.deepEqual(filas.map(f => [f.Tipo, f.Aprobado, f.Intento]), [['Mini evaluación', 'NO', 1], ['Mini evaluación', 'SI', 2]]);

  // La evaluación final del módulo exige todas las lecciones, con sus mini evaluaciones aprobadas.
  const ultimoVideo = v1.lecciones.filter(l => l.miniEvaluacion).pop();
  v1.lecciones.forEach(l => {
    if (l.completada || l.id === ultimoVideo.id) return;
    if (l.miniEvaluacion) env.ejecutar('enviarEvaluacion', 'VENTAS-1', respuestasCorrectas(env, 'VENTAS-1', l.id), l.id);
    else if (!env.ejecutar('obtenerDatos').programa.modulos[1].lecciones.find(x => x.id === l.id).bloqueada) env.ejecutar('marcarLeccion', l.id, true);
  });
  assert.throws(() => env.ejecutar('obtenerEvaluacion', 'VENTAS-1'), /mini evaluaciones/);
  assert.throws(() => env.ejecutar('enviarEvaluacion', 'VENTAS-1', respuestasCorrectas(env, 'VENTAS-1')), /aún no está disponible/);
  env.ejecutar('enviarEvaluacion', 'VENTAS-1', respuestasCorrectas(env, 'VENTAS-1', ultimoVideo.id), ultimoVideo.id);
  env.ejecutar('obtenerDatos').programa.modulos[1].lecciones.filter(l => !l.completada && !l.miniEvaluacion)
    .forEach(l => env.ejecutar('marcarLeccion', l.id, true));
  const final = env.ejecutar('obtenerEvaluacion', 'VENTAS-1');
  assert.equal(final.tipo, 'final');
  assert.equal(final.puntajeAprobacion, 80);
  assert.equal(final.preguntas.length, env.catalogo.preguntas.filter(p => p['ID módulo'] === 'VENTAS-1' && !p['ID lección']).length);
});

test('no se avanza sin responder: lo que sigue a un video se abre al aprobar su mini evaluación', () => {
  const env = entornoInstalado();
  env.comoUsuario(ALUMNA);
  completarModulo(env, 'INDUCCION');
  let v1 = env.ejecutar('obtenerDatos').programa.modulos.find(m => m.id === 'VENTAS-1');
  const orden = v1.lecciones.map(l => l.id);
  assert.equal(orden[0], 'V1-01', 'el módulo parte con su video');
  assert.equal(v1.lecciones[0].bloqueada, false);
  v1.lecciones.slice(1).forEach(l => {
    assert.equal(l.bloqueada, true, l.id + ' bloqueada hasta responder');
    assert.equal(l.bloqueadaPor.id, 'V1-01');
  });
  assert.deepEqual(env.ejecutar('obtenerDatos').programa.siguiente, { moduloId: 'VENTAS-1', leccionId: 'V1-01' });

  // El servidor no deja saltarse las preguntas.
  assert.throws(() => env.ejecutar('marcarLeccion', 'V1-02', true), /primero aprueba la mini evaluación de «TPDV Academy · Ventas \(video\)»/);
  assert.throws(() => env.ejecutar('obtenerEvaluacion', 'VENTAS-1', 'V1-09'), /primero aprueba la mini evaluación/);
  assert.throws(() => env.ejecutar('enviarEvaluacion', 'VENTAS-1', respuestasCorrectas(env, 'VENTAS-1', 'V1-09'), 'V1-09'), /primero aprueba/);
  assert.throws(() => env.ejecutar('obtenerEvaluacion', 'VENTAS-1'), /Completa todas las lecciones/);

  // Reprobar no abre nada; aprobar abre hasta el siguiente video.
  env.ejecutar('enviarEvaluacion', 'VENTAS-1', { 1: 'X', 2: 'X', 3: 'X', 4: 'X' }, 'V1-01');
  v1 = env.ejecutar('obtenerDatos').programa.modulos.find(m => m.id === 'VENTAS-1');
  assert.equal(v1.lecciones.find(l => l.id === 'V1-02').bloqueada, true);
  const r = env.ejecutar('enviarEvaluacion', 'VENTAS-1', respuestasCorrectas(env, 'VENTAS-1', 'V1-01'), 'V1-01');
  v1 = r.datos.programa.modulos.find(m => m.id === 'VENTAS-1');
  const i09 = orden.indexOf('V1-09');
  v1.lecciones.forEach((l, i) => assert.equal(l.bloqueada, i > i09, l.id + (i > i09 ? ' sigue bloqueada (falta V1-09)' : ' abierta')));
  env.ejecutar('marcarLeccion', 'V1-02', true);
  assert.throws(() => env.ejecutar('marcarLeccion', orden[i09 + 2], true), /HubSpot 1/);

  // El siguiente módulo sigue cerrado hasta aprobar la evaluación final.
  assert.equal(r.datos.programa.modulos.find(m => m.id === 'VENTAS-2').bloqueado, true);
  assert.throws(() => env.ejecutar('obtenerEvaluacion', 'VENTAS-2', 'V2-02'), /se desbloquea/);

  // Con ORDEN_OBLIGATORIO = NO la ruta es libre.
  const config = env.hoja('Config');
  config.poner(config.datos.findIndex(f => f[0] === 'ORDEN_OBLIGATORIO') + 1, 2, 'NO');
  const libre = env.ejecutar('obtenerDatos').programa.modulos.find(m => m.id === 'VENTAS-1');
  assert.ok(libre.lecciones.every(l => !l.bloqueada));
  env.ejecutar('marcarLeccion', orden[i09 + 2], true);
});

test('el avance llega a 100% con los cursos y el certificado se puede liberar sin práctica', () => {
  const env = entornoInstalado();
  const config = env.hoja('Config');
  config.poner(config.datos.findIndex(f => f[0] === 'CERTIFICADO_REQUIERE_PRACTICA') + 1, 2, 'NO');
  env.comoUsuario(ALUMNA);
  env.ejecutar('doGet');
  const avances = [];
  ['INDUCCION', 'VENTAS-1', 'VENTAS-2', 'MARKETING-1', 'MARKETING-2', 'LEGAL-1', 'LEGAL-2', 'FINAL'].forEach(id => {
    completarModulo(env, id);
    avances.push(env.ejecutar('obtenerDatos').programa.avance);
  });
  assert.ok(avances.every((a, i) => i === 0 || a > avances[i - 1]), 'el avance sube con cada módulo: ' + avances.join(', '));
  assert.equal(avances[avances.length - 1], 100);
  const datos = env.ejecutar('obtenerDatos');
  assert.equal(datos.certificado.disponible, true, 'sin práctica obligatoria el certificado sale al 100%');
  assert.equal(datos.academia.certificadoRequierePractica, false);
  assert.equal(datos.programa.modulos.find(m => m.id === 'FINAL').practica.requerida, false);
});

test('instalar sobre una versión anterior agrega las mini evaluaciones y la evaluación final que faltan', () => {
  const env = entornoInstalado();
  const hoja = env.hoja('Preguntas');
  const cab = hoja.datos[0];
  const col = cab.indexOf('ID lección');
  const colModulo = cab.indexOf('ID módulo');
  // Versión 1.0: sin columna «ID lección», sin mini evaluaciones y sin preguntas en el módulo FINAL.
  const conservadas = hoja.datos.slice(1).filter(f => !f[col] && f[colModulo] !== 'FINAL').map(f => f.filter((_, j) => j !== col));
  hoja.datos = [cab.filter((_, j) => j !== col)].concat(conservadas);
  const antes = hoja.objetos().length;

  const texto = env.ejecutar('instalar');
  assert.match(texto, /mini evaluaciones/);
  const despues = env.hoja('Preguntas').objetos();
  assert.equal(despues.length, env.catalogo.preguntas.length, 'quedan todas las preguntas del catálogo');
  assert.ok(despues.length > antes);
  assert.equal(despues.filter(p => p['ID lección'] === 'V1-01').length, 4);
  assert.ok(despues.filter(p => p['ID módulo'] === 'FINAL' && !p['ID lección']).length >= 5);
  env.ejecutar('instalar');
  assert.equal(env.hoja('Preguntas').objetos().length, despues.length, 'no se duplican');

  env.comoUsuario(ALUMNA);
  const v1 = env.ejecutar('obtenerDatos').programa.modulos.find(m => m.id === 'VENTAS-1');
  assert.equal(v1.lecciones.find(l => l.id === 'V1-01').miniEvaluacion.preguntas, 4, 'el caché de la estructura se renueva');
});

test('un certificado ya emitido se mantiene aunque cambien las reglas', () => {
  const env = entornoInstalado();
  env.comoUsuario(ALUMNA);
  env.ejecutar('doGet');
  const usuarios = env.hoja('Usuarios');
  const col = usuarios.datos[0].indexOf('Certificado') + 1;
  usuarios.poner(usuarios.datos.findIndex(f => f[0] === ALUMNA) + 1, col, new Date('2026-11-01T12:00:00Z'));
  const datos = env.ejecutar('obtenerDatos');
  assert.equal(datos.programa.terminado, false);
  assert.equal(datos.certificado.disponible, true);
  assert.equal(datos.certificado.esperaPractica, false);
  assert.match(datos.certificado.fecha, /^2026-11-01/);
});

test('desmarcar una lección actualiza el avance y no duplica filas', () => {
  const env = entornoInstalado();
  env.comoUsuario(ALUMNA);
  env.ejecutar('marcarLeccion', 'IND-01', true);
  env.ejecutar('marcarLeccion', 'IND-01', true);
  let datos = env.ejecutar('marcarLeccion', 'IND-01', false);
  assert.equal(datos.programa.leccionesCompletadas, 0);
  assert.equal(env.hoja('Progreso').objetos().length, 1);
  datos = env.ejecutar('marcarLeccion', 'IND-01', true);
  assert.equal(datos.programa.leccionesCompletadas, 1);
});

test('el límite de intentos se respeta cuando está configurado', () => {
  const env = entornoInstalado();
  const config = env.hoja('Config');
  config.poner(config.datos.findIndex(f => f[0] === 'INTENTOS_MAXIMOS') + 1, 2, '2');
  env.comoUsuario(ALUMNA);
  env.ejecutar('obtenerDatos').programa.modulos[0].lecciones.forEach(l => env.ejecutar('marcarLeccion', l.id, true));
  env.ejecutar('enviarEvaluacion', 'INDUCCION', {});
  env.ejecutar('enviarEvaluacion', 'INDUCCION', {});
  assert.throws(() => env.ejecutar('obtenerEvaluacion', 'INDUCCION'), /Usaste los 2 intentos/);
  assert.throws(() => env.ejecutar('enviarEvaluacion', 'INDUCCION', {}), /Usaste los 2 intentos/);
});

test('con ORDEN_OBLIGATORIO = NO todos los módulos quedan abiertos', () => {
  const env = entornoInstalado();
  const config = env.hoja('Config');
  config.poner(config.datos.findIndex(f => f[0] === 'ORDEN_OBLIGATORIO') + 1, 2, 'NO');
  env.comoUsuario(ALUMNA);
  assert.ok(env.ejecutar('obtenerDatos').programa.modulos.every(m => !m.bloqueado));
});

/* ------------------------------------------------------------------ Drive */

test('sincronizar detecta archivos nuevos, sigue atajos a archivos y no recorre atajos a carpetas', () => {
  const env = entornoInstalado();
  const pendientes = env.hoja('Por clasificar').objetos();
  const nuevos = pendientes.filter(p => p['Incluir (SI/NO)'] === '');
  assert.deepEqual(nuevos.map(p => p['ID archivo']), ['archivoNuevoRaiz']);
  assert.equal(nuevos[0].Carpeta, 'TPDV ACADEMY');
  assert.ok(!pendientes.some(p => p['ID archivo'] === 'archivoEnCarpetaExterna'));
  assert.ok(!pendientes.some(p => p['ID archivo'] === 'scriptSuelto'));
  assert.equal(pendientes.filter(p => p['ID archivo'] === '1PLKHyog8kpZGk5MaO6GSz1CnwwFYiucW').length, 1, 'el excluido no se duplica');

  // El administrador lo asigna a un módulo y vuelve a sincronizar.
  const hoja = env.hoja('Por clasificar');
  const filaN = hoja.datos.findIndex(f => f[0] === 'archivoNuevoRaiz') + 1;
  const cab = hoja.datos[0];
  hoja.poner(filaN, cab.indexOf('Asignar a módulo (ID)') + 1, 'VENTAS-2');
  hoja.poner(filaN, cab.indexOf('Incluir (SI/NO)') + 1, 'SI');
  const r = env.ejecutar('sincronizarContenido');
  assert.equal(r.resumen.incorporados, 1);
  assert.equal(r.resumen.nuevos, 0);
  const leccion = env.hoja('Lecciones').objetos().find(l => l['ID archivo'] === 'archivoNuevoRaiz');
  assert.equal(leccion['ID módulo'], 'VENTAS-2');
  assert.equal(leccion['Título'], 'Nuevo instructivo');
  // Una tercera sincronización no lo vuelve a incorporar.
  assert.equal(env.ejecutar('sincronizarContenido').resumen.incorporados, 0);
  env.comoUsuario(ALUMNA);
  const v2 = env.ejecutar('obtenerDatos').programa.modulos.find(m => m.id === 'VENTAS-2');
  assert.ok(v2.lecciones.some(l => l.id === 'D-archivoNuevoRaiz'));
});

test('verificar permisos marca archivos privados o inaccesibles', () => {
  const env = entornoInstalado();
  const lecciones = env.hoja('Lecciones');
  const cab = lecciones.datos[0];
  lecciones.datos.push(cab.map(c => ({ 'ID': 'X-01', 'ID módulo': 'LEGAL-2', 'Título': 'Archivo sin acceso', 'Tipo': 'PDF', 'ID archivo': 'idSinAccesoParaElDueno' }[c] || '')));
  env.contexto.invalidarEstructura_();
  const r = env.ejecutar('verificarPermisos');
  const problemas = Object.fromEntries(r.problemas.map(p => [p.idArchivo, p.problema]));
  assert.match(problemas['1pBt806NPAcvkM7Hh0vJkh_DavV_UWgyy9p0mFjt-bsA'], /No está compartido/);
  assert.match(problemas['idSinAccesoParaElDueno'], /no puede abrir/);
  assert.equal(r.problemas.length, 2);
  assert.ok(r.revisados >= env.catalogo.lecciones.filter(l => l['ID archivo']).length);
});

test('las URLs de los visores se construyen por tipo y no aceptan IDs manipulados', () => {
  const { contexto } = crearEntorno();
  const u = (tipo, idArchivo, url) => JSON.parse(JSON.stringify(contexto.urlsLeccion_({ tipo, idArchivo, url })));
  const id = '1KFvt6BgZGly9oJWTDUyJwtj_V9A2x9ay0AGD7kobfK8';
  assert.equal(u('Documento', id).embed, `https://docs.google.com/document/d/${id}/preview`);
  assert.equal(u('Presentación', id).embed, `https://docs.google.com/presentation/d/${id}/embed?start=false&loop=false`);
  assert.equal(u('Planilla', id).embed, `https://docs.google.com/spreadsheets/d/${id}/preview`);
  assert.equal(u('Video', id).embed, `https://drive.google.com/file/d/${id}/preview`);
  assert.equal(u('PowerPoint', id).embed, `https://drive.google.com/file/d/${id}/preview`);
  assert.equal(u('Carpeta', id).embed, `https://drive.google.com/embeddedfolderview?id=${id}#list`);
  assert.equal(u('Archivo', id).embed, '', 'Keynote y otros formatos solo se abren en Drive');
  assert.equal(u('YouTube', '', 'https://youtu.be/dQw4w9WgXcQ').embed, 'https://www.youtube.com/embed/dQw4w9WgXcQ?rel=0');
  assert.deepEqual(u('PDF', '"><script>alert(1)</script>'), { embed: '', abrir: '' });
  assert.deepEqual(u('Enlace', '', 'javascript:alert(1)'), { embed: '', abrir: '' });
  assert.equal(u('Enlace', '', 'https://tuproyectodevida.cl').abrir, 'https://tuproyectodevida.cl');
});

/* ------------------------------------------------------------------ Marca y seguridad de la página */

test('los valores de marca inválidos en la planilla vuelven al kit oficial', () => {
  const env = entornoInstalado();
  const config = env.hoja('Config');
  const poner = (clave, valor) => config.poner(config.datos.findIndex(f => f[0] === clave) + 1, 2, valor);
  poner('COLOR_PRIMARIO', 'red;} body{display:none');
  poner('FUENTE_TITULOS', "Montserrat'); @import url(evil");
  poner('COLOR_ACENTO', '#abc');
  const marca = env.ejecutar('marca_');
  assert.equal(marca.primario, '#005A82');
  assert.equal(marca.fuenteTitulos, 'Montserrat');
  assert.equal(marca.acento, '#abc');
});

test('un nombre con HTML no rompe la página ni inyecta código', () => {
  const env = entornoInstalado();
  env.comoUsuario(ALUMNA);
  env.ejecutar('guardarPerfil', { nombre: 'Ana </script><script>alert(1)</script>', area: '' });
  // guardarPerfil quita < y >; además se prueba el escape del JSON con un valor crudo en la hoja.
  const usuarios = env.hoja('Usuarios');
  usuarios.poner(usuarios.datos.findIndex(f => f[0] === ALUMNA) + 1, 2, 'Ana </script><script>alert(1)</script>');
  const html = env.ejecutar('doGet').getContent();
  const bloque = html.slice(html.indexOf('window.DATOS_INICIALES'), html.indexOf('</script>', html.indexOf('window.DATOS_INICIALES')));
  assert.ok(bloque.includes('\\u003c/script>'), 'el < queda escapado dentro del JSON');
  assert.ok(!bloque.includes('</script><script>alert'));
});

/* ------------------------------------------------------------------ Integridad del catálogo */

test('el catálogo inicial es consistente', () => {
  const { contexto } = crearEntorno();
  const c = contexto.catalogo_();
  const internos = contexto.contenidoInterno_();
  const idsModulos = new Set(c.modulos.map(m => m.ID));
  assert.equal(idsModulos.size, c.modulos.length, 'IDs de módulo únicos');
  assert.equal(new Set(c.lecciones.map(l => l.ID)).size, c.lecciones.length, 'IDs de lección únicos');
  const tipos = Object.values(contexto.TIPOS_LECCION);

  c.lecciones.forEach(l => {
    assert.ok(idsModulos.has(l['ID módulo']), 'módulo de ' + l.ID);
    assert.ok(tipos.includes(l.Tipo), 'tipo válido en ' + l.ID + ': ' + l.Tipo);
    if (l.Tipo === 'Texto') assert.ok(internos[l.ID], 'contenido interno para ' + l.ID);
    else assert.ok(contexto.idDriveValido_(l['ID archivo']), 'ID de Drive válido en ' + l.ID);
  });
  c.modulos.forEach(m => assert.ok(c.lecciones.some(l => l['ID módulo'] === m.ID), 'el módulo ' + m.ID + ' tiene lecciones'));

  c.preguntas.forEach(p => {
    const ref = p['ID módulo'] + ' #' + p['N°'];
    assert.ok(idsModulos.has(p['ID módulo']), ref);
    assert.ok(['A', 'B', 'C', 'D'].includes(p.Correcta), ref);
    ['A', 'B', 'C', 'D'].forEach(k => assert.ok(String(p[k]).trim(), ref + ' opción ' + k));
    assert.equal(new Set(['A', 'B', 'C', 'D'].map(k => p[k])).size, 4, ref + ' opciones distintas');
    assert.ok(p['Explicación'] && p.Fuente, ref);
  });
  // Cada módulo tiene evaluación final (preguntas sin «ID lección»); el módulo final además es práctico.
  c.modulos.forEach(m =>
    assert.ok(c.preguntas.filter(p => p['ID módulo'] === m.ID && !p['ID lección']).length >= 5, 'al menos 5 preguntas finales en ' + m.ID));
  assert.equal(c.modulos.find(m => m.ID === 'FINAL').Tipo, 'Práctica');
  // Cada video tiene exactamente 4 preguntas de mini evaluación, numeradas 1 a 4 y del mismo módulo.
  const videos = c.lecciones.filter(l => l.Tipo === 'Video');
  assert.ok(videos.length >= 8);
  videos.forEach(v => {
    const mini = c.preguntas.filter(p => p['ID lección'] === v.ID);
    assert.deepEqual(Array.from(mini, p => Number(p['N°'])), [1, 2, 3, 4], 'mini evaluación de ' + v.ID);
    mini.forEach(p => assert.equal(p['ID módulo'], v['ID módulo'], 'módulo de la pregunta de ' + v.ID));
  });
  const idsLeccion = new Set(c.lecciones.map(l => l.ID));
  c.preguntas.filter(p => p['ID lección']).forEach(p => {
    assert.ok(idsLeccion.has(p['ID lección']), 'lección de la pregunta ' + p['ID lección']);
    assert.equal(c.lecciones.find(l => l.ID === p['ID lección']).Tipo, 'Video', 'las mini evaluaciones son de videos');
  });
  const claves = c.preguntas.map(p => p['ID módulo'] + '|' + (p['ID lección'] || '') + '|' + p['N°']);
  assert.equal(new Set(claves).size, claves.length, 'numeración única por evaluación');
  // Las respuestas correctas no se cargan siempre en la misma letra.
  const letras = new Set(c.preguntas.filter(p => p['ID lección']).map(p => p.Correcta));
  assert.ok(letras.size >= 3, 'respuestas correctas repartidas entre letras');

  // Las claves de respuesta del evaluador nunca se publican como lección.
  const archivosPublicados = new Set(c.lecciones.map(l => l['ID archivo']).filter(Boolean));
  ['1PLKHyog8kpZGk5MaO6GSz1CnwwFYiucW', '1RZSvkJrafwRexfvc29objmZIpeeM1Dh3'].forEach(id => assert.ok(!archivosPublicados.has(id), id));
  c.excluidos.forEach(x => assert.ok(!archivosPublicados.has(x['ID archivo']), 'excluido y publicado a la vez: ' + x['ID archivo']));

  // Ningún texto de la plataforma usa «bienvenido/a» con género.
  const textos = JSON.stringify(c) + JSON.stringify(internos);
  assert.doesNotMatch(textos, /bienvenido|¡bienvenid|bienvenid[oa]\/[oa]/i);
});

test('las 4 semanas del programa oficial están representadas en orden', () => {
  const env = entornoInstalado();
  env.comoUsuario(ALUMNA);
  const modulos = env.ejecutar('obtenerDatos').programa.modulos;
  assert.deepEqual(modulos.map(m => m.semana), ['Semana 1', 'Semana 1', 'Semana 2', 'Semana 2', 'Semana 3', 'Semana 3', 'Semana 4', 'Semana 4']);
  assert.deepEqual(modulos.map(m => m.id), ['INDUCCION', 'VENTAS-1', 'VENTAS-2', 'MARKETING-1', 'MARKETING-2', 'LEGAL-1', 'LEGAL-2', 'FINAL']);
});
