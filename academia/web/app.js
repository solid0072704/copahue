/* Plataforma del programa: una sola página con vistas (portada, mi curso, módulo, lección, prueba y certificado).

   DATOS lo escribe sitio.py a partir del programa y de los bancos de preguntas. El avance se guarda en este
   navegador mientras no haya cuentas: `Almacen` es el único punto que cambia cuando se conecte el servicio de
   cuentas. La ruta va en el fragmento de la dirección, con nombres simples (#curso, #modulo-m08, #prueba-m01). */
(function () {
  'use strict';
  var D = window.DATOS, vista = document.getElementById('vista');
  var modPor = {}, lecPor = {};
  D.modulos.forEach(function (m) { modPor[m.clave] = m; m.lecciones.forEach(function (l) { lecPor[l.id] = Object.assign({ modulo: m.clave }, l); }); });

  // ---- Utilidades --------------------------------------------------------------------------------------------
  function esc(t) { return String(t).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function duracion(m) { return m < 60 ? m + ' min' : Math.floor(m / 60) + ' h' + (m % 60 ? ' ' + (m % 60) + ' min' : ''); }
  function horas(m) { return String(Math.round(m / 6) / 10).replace('.', ','); }
  function miles(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, '.'); }
  function fecha(iso) { var d = new Date(iso); return d.getDate() + ' de ' + ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'][d.getMonth()] + ' de ' + d.getFullYear(); }
  function azar(semilla) { var s = (semilla >>> 0) || 1; return function () { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }
  function barajar(lista, r) { var a = lista.slice(); for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)), t = a[i]; a[i] = a[j]; a[j] = t; } return a; }
  function huella(texto) { var h = 2166136261; for (var i = 0; i < texto.length; i++) { h ^= texto.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

  // ---- Avance del alumno -------------------------------------------------------------------------------------
  var Almacen = {
    clave: 'academia-avance-v1',
    leer: function () { try { return JSON.parse(localStorage.getItem(this.clave)) || {}; } catch (e) { return {}; } },
    guardar: function (a) { try { localStorage.setItem(this.clave, JSON.stringify(a)); } catch (e) { /* sin almacenamiento: el avance dura lo que dure la página */ } }
  };
  var avance = Almacen.leer();
  avance.lecciones = avance.lecciones || {};
  avance.pruebas = avance.pruebas || {};
  function guardar() { Almacen.guardar(avance); }
  function mejor(id) { return (avance.pruebas[id] || []).reduce(function (m, x) { return Math.max(m, x.nota); }, -1); }
  function aprobada(id) { return (avance.pruebas[id] || []).some(function (x) { return x.aprobado; }); }
  function tienePrueba(clave) { return !!D.evaluaciones.bancos[clave]; }

  // ---- Piezas comunes ----------------------------------------------------------------------------------------
  function franja(o) {
    var texto = (o.miga ? '<nav class="miga" aria-label="Ubicación">' + o.miga + '</nav>' : '') + '<div class="rotulo">' + o.rotulo + '</div><h1 tabindex="-1">' + o.titulo + '</h1>' +
      (o.bajada ? '<p class="bajada">' + o.bajada + '</p>' : '') + (o.acciones || '');
    return '<header class="franja' + (o.chica ? ' chica' : '') + '"><div class="caja">' + (o.lado ? '<div class="portada"><div>' + texto + '</div>' + o.lado + '</div>' : texto) + (o.extra || '') + '</div></header>';
  }
  function pastillaLeccion(l) {
    if (avance.lecciones[l.id] && avance.lecciones[l.id].hecha) return '<span class="pastilla ok">Vista</span>';
    return l.video ? '<span class="pastilla oro">Disponible</span>' : '<span class="pastilla">En producción</span>';
  }
  function tipoDe(l) { return { clase: 'Clase', demo: 'Demostración', laboratorio: 'Laboratorio' }[l.tipo]; }
  function listaLecciones(m, enlaces) {
    return '<ol class="lecciones">' + m.lecciones.map(function (l) {
      var cuerpo = '<b>' + l.n + '. ' + esc(l.t) + '</b><span class="min">' + tipoDe(l) + ' · ' + l.min + ' min</span><span>' + esc(l.d) + '</span>';
      return '<li>' + (enlaces ? '<a href="#leccion-' + l.id + '" style="display:contents">' + cuerpo + '</a>' : cuerpo) + '</li>';
    }).join('') + '</ol>';
  }

  // ---- Portada -----------------------------------------------------------------------------------------------
  function programaHtml() {
    return D.bloques.map(function (b) {
      var ms = D.modulos.filter(function (m) { return m.bloque === b.clave; });
      return '<section class="bloque"><header><h3>' + esc(b.t) + '</h3><span class="rotulo">Etapa ' + b.etapa + ' · ' + ms.length + ' módulos · ' + horas(ms.reduce(function (s, m) { return s + m.min; }, 0)) + ' h</span></header>' +
        '<p>' + esc(b.d) + '</p>' + ms.map(function (m) {
          return '<details class="mod"><summary><span class="n">' + ('0' + m.n).slice(-2) + '</span><b>' + esc(m.t) + '</b><span class="meta">' + m.lecciones.length + ' lecciones · ' + duracion(m.min) + '</span>' +
            '<span class="d">' + esc(m.d) + '</span></summary><div class="dentro">' + listaLecciones(m, false) +
            '<div class="practica"><b>Práctica</b>' + esc(m.practica) + '</div></div></details>';
        }).join('') + '</section>';
    }).join('');
  }
  function vistaInicio(ancla) {
    var t = D.totales, e1 = D.porEtapa[1], r = D.evaluaciones.reglas;
    vista.innerHTML = franja({
      rotulo: 'Programa profesional · en línea, a tu ritmo',
      titulo: 'Claude para Finanzas <em>y Contabilidad</em>',
      bajada: 'Aprende cómo funciona Claude y úsalo en serio: con tus planillas, tu CRM y Power BI, en el cierre, el presupuesto y los informes. Con evaluaciones, casos prácticos y certificado.',
      acciones: '<div class="acciones"><a class="btn pri" href="#programa">Ver el programa</a>' + (D.muestra ? '<a class="btn sec" href="#muestra">Probar una lección</a>' : '') + '</div>',
      lado: D.muestra ? '<a class="ver-muestra" href="#muestra" aria-label="Ver la lección de muestra"><img src="' + D.muestra.poster + '" alt="" width="1280" height="720"><span class="play"></span>' +
        '<span class="pie-muestra">Lección de muestra · ' + D.muestra.min + ' min</span></a>' : '',
      extra: '<div class="cifras"><div><b>' + t.modulos + '</b><span>módulos</span></div><div><b>' + t.lecciones + '</b><span>lecciones narradas</span></div>' +
        '<div><b>' + horas(t.minutos) + ' h</b><span>de clases</span></div><div><b>' + (t.modulos + 2) + '</b><span>evaluaciones</span></div></div>'
    }) +
    '<section class="seccion"><div class="caja"><div class="rotulo">Lo que vas a poder hacer</div><h2>Del «pregúntale a la IA» a un <em>método de trabajo</em></h2>' +
      '<ul class="lista-logros">' + D.logros.map(function (x) { return '<li><span>' + esc(x) + '</span></li>'; }).join('') + '</ul></div></section>' +
    '<section class="seccion" id="programa"><div class="caja"><div class="rotulo">Programa</div><h2>' + t.modulos + ' módulos, de los fundamentos a la <em>automatización</em></h2>' +
      '<p class="entrada">Cada módulo tiene lecciones cortas, una práctica con datos y una prueba. Abre un módulo para ver sus lecciones.</p>' + programaHtml() + '</div></section>' +
    '<section class="seccion" id="como-se-evalua"><div class="caja"><div class="rotulo">Evaluación</div><h2>Se aprende haciendo, y <em>se comprueba</em></h2>' +
      '<ol class="pasos"><li><b>Diagnóstico inicial</b><span>Diez preguntas para saber por dónde te conviene partir.</span></li>' +
      '<li><b>Una prueba por módulo</b><span>' + r.preguntas_por_modulo + ' preguntas tomadas de un banco más grande. Se aprueba con ' + r.aprobacion + ' % y cada intento trae preguntas distintas.</span></li>' +
      '<li><b>Laboratorios con datos</b><span>Ejercicios con el caso Distribuidora Alerce: libro mayor, CRM y modelo de Power BI, con respuesta numérica.</span></li>' +
      '<li><b>Examen final y certificado</b><span>' + r.examen_final + ' preguntas de todo el programa. Con ' + r.aprobacion_final + ' % y las pruebas aprobadas, obtienes el certificado.</span></li></ol>' +
      '<div class="acciones" style="display:flex;flex-wrap:wrap;gap:12px;margin-top:22px"><a class="btn pri" href="#prueba-diagnostico">Hacer el diagnóstico</a><a class="btn sec" href="#prueba-m01">Ver la prueba del módulo 1</a></div></div></section>' +
    '<section class="seccion"><div class="caja"><div class="rotulo">Quién enseña</div><div class="autor"><img src="' + D.autor.foto + '" alt="Retrato de ' + esc(D.autor.nombre) + '" width="200" height="200">' +
      '<div><h3>' + esc(D.autor.nombre) + '</h3><p>' + esc(D.autor.titulo) + '.</p><p>' + esc(D.autor.bio) + '</p></div></div></div></section>' +
    '<section class="seccion" id="inscripcion"><div class="caja"><div class="rotulo">Inscripción</div><h2>El programa se publica <em>por etapas</em></h2>' +
      '<p class="entrada">La primera etapa cubre cómo funciona Claude, sus herramientas y las conexiones con CRM y Power BI. Las siguientes se agregan al programa a medida que se publican.</p>' +
      '<div class="desliza"><table class="tabla-etapas"><thead><tr><th>Etapa</th><th>Contenido</th><th class="num">Módulos</th><th class="num">Lecciones</th><th class="num">Horas</th></tr></thead><tbody>' +
      Object.keys(D.etapas).map(function (k) { var x = D.porEtapa[k]; return '<tr><td>' + k + '</td><td>' + esc(D.etapas[k]) + '</td><td class="num">' + x.modulos + '</td><td class="num">' + x.lecciones + '</td><td class="num">' + horas(x.minutos) + '</td></tr>'; }).join('') +
      '</tbody></table></div><div class="dos par"><div class="panel"><h3>Qué incluye</h3><p>Las ' + t.lecciones + ' lecciones narradas, los archivos del caso, los prompts y plantillas de cada módulo, las evaluaciones y el certificado.</p></div>' +
      '<div class="panel"><h3>Precio y fecha de inicio</h3><p>Por definir. La etapa 1 suma ' + e1.modulos + ' módulos y ' + horas(e1.minutos) + ' horas de clases.</p></div></div></div></section>' +
    '<section class="seccion faq"><div class="caja"><div class="rotulo">Preguntas frecuentes</div>' + D.preguntas.map(function (x) { return '<details><summary>' + esc(x[0]) + '</summary><p>' + esc(x[1]) + '</p></details>'; }).join('') + '</div></section>';
    if (ancla && ancla !== 'inicio') { var destino = document.getElementById(ancla); if (destino) { destino.scrollIntoView(); return true; } }
  }

  // ---- Mi curso ----------------------------------------------------------------------------------------------
  function vistaCurso() {
    var hechas = Object.keys(avance.lecciones).filter(function (k) { return avance.lecciones[k].hecha; }).length;
    var publicadas = Object.keys(lecPor).filter(function (k) { return lecPor[k].video; }).length + (D.muestra ? 1 : 0);
    var conPrueba = D.modulos.filter(function (m) { return tienePrueba(m.clave); }), pasadas = conPrueba.filter(function (m) { return aprobada(m.clave); }).length;
    var diag = avance.pruebas.diagnostico && avance.pruebas.diagnostico.slice(-1)[0];
    vista.innerHTML = franja({
      chica: true, rotulo: 'Mi curso', titulo: 'Tu avance en el programa',
      bajada: hechas + ' de ' + publicadas + ' lecciones publicadas vistas · ' + pasadas + ' de ' + conPrueba.length + ' pruebas publicadas aprobadas',
      extra: '<div class="avance" style="margin-top:18px;max-width:520px;--p:' + (publicadas ? Math.round(100 * hechas / publicadas) : 0) + '"><i></i></div>'
    }) + '<section class="seccion"><div class="caja"><div class="dos par" style="margin-top:0">' +
      '<div class="panel"><h3>Diagnóstico inicial</h3><p>' + (diag ? 'Lo rendiste el ' + fecha(diag.fecha) + ': ' + diag.buenas + ' de ' + diag.total + ' correctas. ' + esc(recomendacion(diag.buenas)) : 'Diez preguntas para saber por dónde te conviene partir. No lleva nota.') + '</p>' +
      '<a class="btn ' + (diag ? 'sec' : 'pri') + '" href="#prueba-diagnostico">' + (diag ? 'Repetir el diagnóstico' : 'Hacer el diagnóstico') + '</a></div>' +
      (D.muestra ? '<div class="panel"><h3>Lección de muestra</h3><p>' + esc(D.muestra.t) + ' · ' + D.muestra.min + ' min.</p><a class="btn sec" href="#muestra">Ver la lección</a></div>' : '') + '</div>' +
      D.bloques.map(function (b) {
        return '<section class="bloque"><header><h3>' + esc(b.t) + '</h3><span class="rotulo">Etapa ' + b.etapa + '</span></header><div class="rejilla">' +
          D.modulos.filter(function (m) { return m.bloque === b.clave; }).map(function (m) {
            var listas = m.lecciones.filter(function (l) { return l.video; }).length;
            return '<a class="tarjeta" href="#modulo-' + m.clave + '"><span class="n">Módulo ' + m.n + '</span><b>' + esc(m.t) + '</b><span class="d">' + esc(m.d) + '</span><span class="abajo">' +
              (listas ? '<span class="pastilla oro">' + listas + ' de ' + m.lecciones.length + ' lecciones</span>' : '<span class="pastilla">En producción</span>') +
              (tienePrueba(m.clave) ? (aprobada(m.clave) ? '<span class="pastilla ok">Prueba aprobada</span>' : '<span class="pastilla oro">Prueba disponible</span>') : '') +
              '<span class="min">' + duracion(m.min) + '</span></span></a>';
          }).join('') + '</div></section>';
      }).join('') +
      '<section class="bloque"><header><h3>Certificado</h3></header><p>Se obtiene al aprobar las pruebas de los módulos y el examen final.</p><p style="margin-top:14px"><a class="btn sec" href="#certificado">Ver los requisitos</a></p></section></div></section>';
  }

  // ---- Módulo ------------------------------------------------------------------------------------------------
  function vistaModulo(m) {
    var b = D.bloques.filter(function (x) { return x.clave === m.bloque; })[0], id = m.clave, nota = mejor(id);
    vista.innerHTML = franja({
      chica: true, miga: '<a href="#curso">Mi curso</a><span>›</span><span>' + esc(b.t) + '</span>', rotulo: 'Módulo ' + m.n + ' · ' + m.lecciones.length + ' lecciones · ' + duracion(m.min),
      titulo: esc(m.t), bajada: esc(m.d)
    }) + '<section class="seccion"><div class="caja"><div class="dos" style="margin-top:0"><div><div class="rotulo" style="color:var(--acento)">Lecciones</div>' + listaLeccionesConEstado(m) +
      '<div class="practica"><b>Práctica del módulo</b>' + esc(m.practica) + '</div></div>' +
      '<aside><div class="panel"><h3>Al terminar vas a poder</h3><ul class="texto" style="margin-top:6px"><ul>' + m.logras.map(function (x) { return '<li>' + esc(x) + '</li>'; }).join('') + '</ul></ul></div>' +
      '<div class="panel" style="margin-top:16px"><h3>Prueba del módulo</h3>' + (tienePrueba(id) ?
        '<p>' + Math.min(D.evaluaciones.reglas.preguntas_por_modulo, D.evaluaciones.bancos[id].length) + ' preguntas. Se aprueba con ' + D.evaluaciones.reglas.aprobacion + ' %.' + (nota >= 0 ? ' Tu mejor nota: ' + nota + ' %.' : '') + '</p>' +
        '<a class="btn pri" href="#prueba-' + id + '">' + (nota >= 0 ? 'Rendirla de nuevo' : 'Rendir la prueba') + '</a>' :
        '<p>Se publica junto con las lecciones del módulo.</p>') + '</div></aside></div></div></section>';
  }
  function listaLeccionesConEstado(m) {
    return '<ol class="lecciones">' + m.lecciones.map(function (l) {
      return '<li><a href="#leccion-' + l.id + '" style="display:contents"><b>' + l.n + '. ' + esc(l.t) + '</b><span class="min">' + tipoDe(l) + ' · ' + l.min + ' min</span><span>' + esc(l.d) + '</span></a>' +
        '<span style="margin-top:4px">' + pastillaLeccion(l) + '</span></li>';
    }).join('') + '</ol>';
  }

  // ---- Lección -----------------------------------------------------------------------------------------------
  function reproductor(id, medio) {
    return '<div class="reproductor"><video id="video" controls playsinline preload="metadata" poster="' + medio.poster + '" src="' + medio.video + '">' +
      (medio.vtt ? '<track kind="subtitles" srclang="es" label="Español" src="' + medio.vtt + '" default>' : '') + '</video></div>';
  }
  function enlazarVideo(id, medio) {
    var v = document.getElementById('video'), estado = avance.lecciones[id] || (avance.lecciones[id] = {}), ultimo = 0;
    if (!v) return;
    if (medio.subtitulos && v.addTextTrack && window.VTTCue) {       // subtítulos que vienen en los datos y no en un archivo
      var pista = v.addTextTrack('subtitles', 'Español', 'es');
      medio.subtitulos.forEach(function (c) { pista.addCue(new VTTCue(c[0], c[1], c[2])); });
      pista.mode = 'showing';
    }
    v.addEventListener('loadedmetadata', function () { if (estado.t && estado.t < v.duration - 5) v.currentTime = estado.t; });
    v.addEventListener('timeupdate', function () {
      if (Math.abs(v.currentTime - ultimo) < 5) return;
      ultimo = estado.t = Math.round(v.currentTime);
      if (v.duration && v.currentTime > v.duration * .9 && !estado.hecha) { estado.hecha = true; pintarEstadoLeccion(id); }
      guardar();
    });
  }
  function pintarEstadoLeccion(id) {
    var b = document.getElementById('marcar'), hecha = avance.lecciones[id] && avance.lecciones[id].hecha;
    if (b) { b.textContent = hecha ? 'Vista ✓ (desmarcar)' : 'Marcar como vista'; b.setAttribute('aria-pressed', hecha ? 'true' : 'false'); }
  }
  function enlazarMarcar(id) {
    var b = document.getElementById('marcar');
    if (!b) return;
    b.addEventListener('click', function () { var e = avance.lecciones[id] || (avance.lecciones[id] = {}); e.hecha = !e.hecha; guardar(); pintarEstadoLeccion(id); });
    pintarEstadoLeccion(id);
  }
  function vistaLeccion(l) {
    var m = modPor[l.modulo], i = m.lecciones.findIndex(function (x) { return x.id === l.id; }), ant = m.lecciones[i - 1], sig = m.lecciones[i + 1];
    vista.innerHTML = franja({
      chica: true, miga: '<a href="#curso">Mi curso</a><span>›</span><a href="#modulo-' + m.clave + '">Módulo ' + m.n + '</a>',
      rotulo: tipoDe(l) + ' ' + m.n + '.' + l.n + ' · ' + l.min + ' min', titulo: esc(l.t), bajada: esc(l.d)
    }) + '<section class="seccion"><div class="caja"><div class="dos" style="margin-top:0"><div>' +
      (l.video ? reproductor(l.id, l) : '<div class="en-produccion"><div><b>Esta lección está en producción</b><span>Se publica con el módulo ' + m.n + '. Mientras tanto puedes ver la lección de muestra o rendir el diagnóstico.</span></div></div>') +
      '<div class="bajo-video">' + (l.video ? '<button class="btn liso" id="marcar" type="button"></button>' : '<a class="btn liso" href="#muestra">Ver la lección de muestra</a>') + '<span class="espacio"></span>' +
      (ant ? '<a class="btn liso" href="#leccion-' + ant.id + '">← Anterior</a>' : '') + (sig ? '<a class="btn liso" href="#leccion-' + sig.id + '">Siguiente →</a>' : (tienePrueba(m.clave) ? '<a class="btn pri" href="#prueba-' + m.clave + '">Ir a la prueba</a>' : '')) + '</div>' +
      '<div class="texto"><h3>En este módulo vas a poder</h3><ul>' + m.logras.map(function (x) { return '<li>' + esc(x) + '</li>'; }).join('') + '</ul></div></div>' +
      '<aside class="lateral"><h4>Módulo ' + m.n + ' · ' + esc(m.t) + '</h4><ol>' + m.lecciones.map(function (x) {
        return '<li><a href="#leccion-' + x.id + '"' + (x.id === l.id ? ' aria-current="page"' : '') + '><span class="n">' + x.n + '</span><span>' + esc(x.t) + '</span></a></li>';
      }).join('') + '</ol></aside></div></div></section>';
    if (l.video) { enlazarVideo(l.id, l); enlazarMarcar(l.id); }
  }
  function vistaMuestra() {
    var x = D.muestra;
    if (!x) { location.hash = 'inicio'; return; }
    vista.innerHTML = franja({ chica: true, miga: '<a href="#inicio">Portada</a>', rotulo: 'Lección de muestra · ' + x.min + ' min', titulo: esc(x.t), bajada: esc(x.d) }) +
      '<section class="seccion"><div class="caja"><div class="dos" style="margin-top:0"><div>' + reproductor('muestra', x) +
      '<div class="bajo-video"><button class="btn liso" id="marcar" type="button"></button><span class="espacio"></span><a class="btn pri" href="#programa">Ver el programa completo</a></div>' +
      '<div class="texto"><h3>Qué vas a ver</h3><ul>' + x.puntos.map(function (p) { return '<li>' + esc(p) + '</li>'; }).join('') + '</ul><p>' + esc(x.nota) + '</p></div></div>' +
      '<aside><div class="panel"><h3>Así son las lecciones</h3><p>Entre 5 y 9 minutos, narradas, con subtítulos y con el avance guardado: al volver, sigues donde quedaste.</p></div>' +
      '<div class="panel" style="margin-top:16px"><h3>¿Y después?</h3><p>Cada módulo cierra con una práctica y una prueba.</p><a class="btn sec" href="#prueba-m01">Ver una prueba</a></div></aside></div></div></section>';
    enlazarVideo('muestra', x); enlazarMarcar('muestra');
  }

  // ---- Pruebas -----------------------------------------------------------------------------------------------
  var prueba = null;                // la prueba que se está rindiendo
  function recomendacion(buenas) { var r = ''; D.evaluaciones.recomendaciones.forEach(function (x) { if (buenas >= x[0]) r = x[1]; }); return r; }
  function armar(id) {
    var esDiag = id === 'diagnostico', banco = esDiag ? D.evaluaciones.diagnostico : D.evaluaciones.bancos[id];
    var intento = (avance.pruebas[id] || []).length + 1, r = azar(huella(id) + intento * 7919);
    var elegidas = esDiag ? banco.slice() : barajar(banco, r).slice(0, D.evaluaciones.reglas.preguntas_por_modulo);
    return {
      id: id, i: 0, entregada: false, intento: intento,
      preguntas: elegidas.map(function (q) {
        var orden = q.o.map(function (_, k) { return k; });
        if (q.tipo === 'orden') { do { orden = barajar(orden, r); } while (orden.every(function (k, j) { return k === j; })); }
        else if (q.tipo !== 'numero') orden = barajar(orden, r);
        return { q: q, orden: orden, r: q.tipo === 'multiple' ? [] : q.tipo === 'orden' ? orden.slice() : null };
      })
    };
  }
  function correcta(x) {
    var q = x.q;
    if (q.tipo === 'unica') return x.r === q.r;
    if (q.tipo === 'multiple') return x.r.length === q.r.length && q.r.every(function (k) { return x.r.indexOf(k) >= 0; });
    if (q.tipo === 'numero') { var n = parseFloat(String(x.r == null ? '' : x.r).replace(/\./g, '').replace(',', '.').replace(/[^0-9.\-]/g, '')); return !isNaN(n) && Math.abs(n - q.r) <= (q.tol || 0) + 1e-9; }
    return x.r.every(function (k, j) { return k === j; });
  }
  function respondida(x) { return x.q.tipo === 'orden' || (x.q.tipo === 'multiple' ? x.r.length > 0 : x.r !== null && x.r !== ''); }
  function nombrePrueba(id) { return id === 'diagnostico' ? 'Diagnóstico inicial' : 'Prueba del módulo ' + modPor[id].n; }
  function vistaPrueba(id) {
    if (id !== 'diagnostico' && !tienePrueba(id)) {
      vista.innerHTML = franja({ chica: true, miga: '<a href="#curso">Mi curso</a>', rotulo: 'Evaluación', titulo: 'Esta prueba todavía no está publicada', bajada: 'Las pruebas se publican junto con las lecciones de su módulo.' }) +
        '<section class="seccion"><div class="caja"><a class="btn pri" href="#prueba-m01">Ver la prueba del módulo 1</a></div></section>';
      return;
    }
    if (!prueba || prueba.id !== id) prueba = null;
    var esDiag = id === 'diagnostico', reglas = D.evaluaciones.reglas, banco = esDiag ? D.evaluaciones.diagnostico : D.evaluaciones.bancos[id];
    var cabeza = franja({
      chica: true, miga: '<a href="#curso">Mi curso</a>' + (esDiag ? '' : '<span>›</span><a href="#modulo-' + id + '">Módulo ' + modPor[id].n + '</a>'), rotulo: 'Evaluación',
      titulo: nombrePrueba(id), bajada: esDiag ? 'No lleva nota: sirve para recomendarte por dónde partir.' : esc(modPor[id].t)
    });
    if (!prueba) {
      var intentos = avance.pruebas[id] || [], n = esDiag ? banco.length : Math.min(reglas.preguntas_por_modulo, banco.length);
      vista.innerHTML = cabeza + '<section class="seccion"><div class="caja"><div class="prueba"><div class="panel"><h3>Antes de empezar</h3>' +
        '<p>' + n + ' preguntas' + (esDiag ? '.' : ', tomadas de un banco de ' + banco.length + '. Se aprueba con ' + reglas.aprobacion + ' %; puedes repetirla y cada intento trae preguntas distintas.') +
        ' Al entregar ves la respuesta correcta y su explicación.</p>' +
        (intentos.length ? '<p>Intentos anteriores: ' + intentos.map(function (x) { return esDiag ? x.buenas + ' de ' + x.total : x.nota + ' %'; }).join(' · ') + '.</p>' : '') +
        '<button class="btn pri" id="empezar" type="button">Empezar</button></div></div></div></section>';
      document.getElementById('empezar').addEventListener('click', function () { prueba = armar(id); vistaPrueba(id); });
      return;
    }
    vista.innerHTML = cabeza + '<section class="seccion"><div class="caja"><div class="prueba" id="zona"></div></div></section>';
    if (prueba.entregada) pintarResultado(); else pintarPregunta();
  }
  function pintarPregunta() {
    var zona = document.getElementById('zona'), x = prueba.preguntas[prueba.i], q = x.q, total = prueba.preguntas.length, cuerpo;
    if (q.tipo === 'unica' || q.tipo === 'multiple') {
      cuerpo = '<div class="opciones" role="' + (q.tipo === 'unica' ? 'radiogroup' : 'group') + '" aria-labelledby="enunciado">' + x.orden.map(function (k, j) {
        var marcada = q.tipo === 'unica' ? x.r === k : x.r.indexOf(k) >= 0;
        return '<label class="opcion"><input type="' + (q.tipo === 'unica' ? 'radio' : 'checkbox') + '" name="op" id="op-' + j + '" value="' + k + '"' + (marcada ? ' checked' : '') + '><span>' + esc(q.o[k]) + '</span></label>';
      }).join('') + '</div>';
    } else if (q.tipo === 'numero') {
      cuerpo = '<div class="campo"><label for="numero" class="rotulo">Tu respuesta</label><input id="numero" inputmode="decimal" autocomplete="off" value="' + esc(x.r == null ? '' : x.r) + '">' + (q.unidad ? '<span>' + esc(q.unidad) + '</span>' : '') + '</div>';
    } else {
      cuerpo = '<ol class="ordenable">' + x.r.map(function (k, j) {
        return '<li><span>' + esc(q.o[k]) + '</span><button type="button" data-sube="' + j + '" aria-label="Subir"' + (j ? '' : ' disabled') + '>↑</button><button type="button" data-baja="' + j + '" aria-label="Bajar"' + (j < x.r.length - 1 ? '' : ' disabled') + '>↓</button></li>';
      }).join('') + '</ol>';
    }
    zona.innerHTML = '<div class="cabeza"><span class="num">Pregunta ' + (prueba.i + 1) + ' de ' + total + '</span><div class="avance" style="--p:' + Math.round(100 * prueba.i / total) + '"><i></i></div></div>' +
      '<p class="pregunta" id="enunciado">' + esc(q.p) + '</p>' + (q.tipo === 'multiple' ? '<p class="ayuda">Puede haber más de una correcta.</p>' : q.tipo === 'orden' ? '<p class="ayuda">Ordena con las flechas.</p>' : '') + cuerpo +
      '<div class="pie-prueba"><button class="btn liso" id="atras" type="button"' + (prueba.i ? '' : ' disabled') + '>← Anterior</button>' +
      '<button class="btn pri" id="sigue" type="button">' + (prueba.i === total - 1 ? 'Entregar' : 'Siguiente →') + '</button></div><p class="ayuda" id="falta" role="status"></p>';
    zona.querySelectorAll('input[name=op]').forEach(function (el) {
      el.addEventListener('change', function () {
        var k = +el.value;
        if (q.tipo === 'unica') x.r = k; else if (el.checked) x.r.push(k); else x.r = x.r.filter(function (v) { return v !== k; });
      });
    });
    var campo = document.getElementById('numero');
    if (campo) campo.addEventListener('input', function () { x.r = campo.value.trim(); });
    zona.querySelectorAll('[data-sube],[data-baja]').forEach(function (b) {
      b.addEventListener('click', function () {
        var j = +(b.dataset.sube || b.dataset.baja), d = b.dataset.sube !== undefined ? -1 : 1, t = x.r[j];
        x.r[j] = x.r[j + d]; x.r[j + d] = t; pintarPregunta();
      });
    });
    document.getElementById('atras').addEventListener('click', function () { prueba.i--; pintarPregunta(); });
    document.getElementById('sigue').addEventListener('click', function () {
      if (!respondida(x)) { document.getElementById('falta').textContent = 'Responde esta pregunta para continuar.'; return; }
      if (prueba.i < total - 1) { prueba.i++; pintarPregunta(); } else entregar();
    });
  }
  function entregar() {
    var buenas = prueba.preguntas.filter(correcta).length, total = prueba.preguntas.length, nota = Math.round(100 * buenas / total), esDiag = prueba.id === 'diagnostico';
    prueba.entregada = true; prueba.buenas = buenas; prueba.nota = nota; prueba.aprobado = !esDiag && nota >= D.evaluaciones.reglas.aprobacion;
    (avance.pruebas[prueba.id] = avance.pruebas[prueba.id] || []).push({ fecha: new Date().toISOString(), nota: nota, buenas: buenas, total: total, aprobado: prueba.aprobado });
    guardar(); pintarResultado(); window.scrollTo(0, 0);
  }
  function textoRespuesta(x, deAlumno) {
    var q = x.q;
    if (q.tipo === 'unica') return deAlumno ? q.o[x.r] : q.o[q.r];
    if (q.tipo === 'multiple') return (deAlumno ? x.r.slice().sort() : q.r).map(function (k) { return q.o[k]; }).join(' · ');
    if (q.tipo === 'numero') return (deAlumno ? x.r : String(q.r).replace('.', ',')) + (q.unidad ? ' ' + q.unidad : '');
    return (deAlumno ? x.r : q.o.map(function (_, k) { return k; })).map(function (k, j) { return (j + 1) + ') ' + q.o[k]; }).join('  ');
  }
  function pintarResultado() {
    var zona = document.getElementById('zona'), esDiag = prueba.id === 'diagnostico', total = prueba.preguntas.length, reglas = D.evaluaciones.reglas, mensaje, extra = '';
    if (esDiag) {
      var temas = {};
      prueba.preguntas.forEach(function (x) { if (!correcta(x)) temas[x.q.tema] = 1; });
      mensaje = recomendacion(prueba.buenas);
      extra = Object.keys(temas).length ? '<p style="margin-top:10px">Para reforzar: ' + Object.keys(temas).sort().map(function (k) { return '<a href="#modulo-' + k + '">módulo ' + modPor[k].n + '</a>'; }).join(', ') + '.</p>' : '';
    } else {
      mensaje = prueba.aprobado ? 'Aprobaste la prueba del módulo ' + modPor[prueba.id].n + '.' : 'Te faltó para el ' + reglas.aprobacion + ' %. Revisa las explicaciones y vuelve a intentarlo: el próximo intento trae otras preguntas.';
    }
    zona.innerHTML = '<div class="panel"><div class="nota"><div class="aro" style="--p:' + prueba.nota + ';--c:' + (esDiag || prueba.aprobado ? 'var(--verde)' : 'var(--rojo)') + '"><b>' + (esDiag ? prueba.buenas + '/' + total : prueba.nota + ' %') + '</b></div>' +
      '<div><h3>' + (esDiag ? 'Tu punto de partida' : prueba.aprobado ? 'Aprobada' : 'No aprobada') + '</h3><p>' + prueba.buenas + ' de ' + total + ' correctas. ' + esc(mensaje) + '</p>' + extra +
      '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:16px"><button class="btn sec" id="otra" type="button">' + (esDiag ? 'Repetir' : 'Intentar de nuevo') + '</button><a class="btn pri" href="#curso">Ir a mi curso</a></div></div></div></div>' +
      '<ol class="revision">' + prueba.preguntas.map(function (x, j) {
        var ok = correcta(x), l = x.q.l && lecPor[x.q.l];
        return '<li class="' + (ok ? '' : 'mal') + '"><b>' + (j + 1) + '. ' + esc(x.q.p) + '</b>' +
          '<div class="tuya"><span class="pastilla ' + (ok ? 'ok' : 'mal') + '">' + (ok ? 'Correcta' : 'Incorrecta') + '</span> Tu respuesta: ' + esc(textoRespuesta(x, true)) + '</div>' +
          (ok ? '' : '<div class="buena">Respuesta correcta: <b style="display:inline">' + esc(textoRespuesta(x, false)) + '</b></div>') +
          '<div class="exp">' + esc(x.q.e) + (l ? ' <a href="#leccion-' + l.id + '">Lección ' + modPor[l.modulo].n + '.' + l.n + '</a>' : '') + '</div></li>';
      }).join('') + '</ol>';
    document.getElementById('otra').addEventListener('click', function () { var id = prueba.id; prueba = armar(id); vistaPrueba(id); window.scrollTo(0, 0); });
  }

  // ---- Certificado -------------------------------------------------------------------------------------------
  function vistaCertificado() {
    var conPrueba = D.modulos.filter(function (m) { return tienePrueba(m.clave); });
    function diploma() {
      var nombre = (avance.nombre || '').trim() || 'Tu nombre', codigo = ('0000000' + huella(nombre + D.nombre).toString(36)).slice(-7).toUpperCase();
      return '<div class="diploma"><span class="sello">Muestra</span><div class="rotulo">Certificado de aprobación</div><h3>' + esc(nombre) + '</h3>' +
        '<p>aprobó el programa <b>' + esc(D.nombre) + '</b>: ' + D.totales.modulos + ' módulos y ' + horas(D.totales.minutos) + ' horas de clases, con sus evaluaciones y el examen final.</p>' +
        '<div class="firma"><div><b>' + esc(D.autor.nombre) + '</b>' + esc(D.autor.titulo) + '</div><div><b>Código de verificación</b><span class="num">' + codigo + '</span></div></div></div>';
    }
    vista.innerHTML = franja({ chica: true, miga: '<a href="#curso">Mi curso</a>', rotulo: 'Certificado', titulo: 'Requisitos y vista del certificado', bajada: 'Se emite al aprobar las pruebas de todos los módulos y el examen final.' }) +
      '<section class="seccion"><div class="caja"><div class="dos" style="margin-top:0"><div><div id="diploma">' + diploma() + '</div>' +
      '<div class="campo" style="flex-wrap:wrap"><label for="nombre" class="rotulo">Nombre en el certificado</label><input id="nombre" style="width:20rem;font:600 1rem var(--f-texto)" autocomplete="name" value="' + esc(avance.nombre || '') + '"></div></div>' +
      '<aside><div class="rotulo" style="color:var(--acento)">Requisitos</div><ul class="requisitos" style="margin-top:8px">' +
      conPrueba.map(function (m) { return '<li class="' + (aprobada(m.clave) ? 'ok' : '') + '"><span>Prueba del módulo ' + m.n + (aprobada(m.clave) ? ' · aprobada con ' + mejor(m.clave) + ' %' : '') + '</span></li>'; }).join('') +
      '<li><span>Pruebas de los módulos ' + (conPrueba.length + 1) + ' a ' + D.modulos.length + ' (se publican con cada módulo)</span></li><li><span>Examen final: ' + D.evaluaciones.reglas.examen_final + ' preguntas, con ' + D.evaluaciones.reglas.aprobacion_final + ' % o más</span></li></ul></aside></div></div></section>';
    document.getElementById('nombre').addEventListener('input', function (e) { avance.nombre = e.target.value; guardar(); document.getElementById('diploma').innerHTML = diploma(); });
  }

  // ---- Rutas -------------------------------------------------------------------------------------------------
  function pintar() {
    var r = (location.hash || '#inicio').slice(1), m, quieto = false;
    if (r === 'curso') vistaCurso();
    else if ((m = /^modulo-(m\d\d)$/.exec(r)) && modPor[m[1]]) vistaModulo(modPor[m[1]]);
    else if ((m = /^leccion-(m\d\d-l\d\d)$/.exec(r)) && lecPor[m[1]]) vistaLeccion(lecPor[m[1]]);
    else if (r === 'muestra') vistaMuestra();
    else if ((m = /^prueba-(diagnostico|m\d\d)$/.exec(r)) && (m[1] === 'diagnostico' || modPor[m[1]])) vistaPrueba(m[1]);
    else if (r === 'certificado') vistaCertificado();
    else quieto = vistaInicio(r);
    document.querySelectorAll('.barra nav a').forEach(function (a) {
      var esta = a.getAttribute('href') === '#' + r || (a.getAttribute('href') === '#curso' && /^(modulo|leccion|prueba|certificado)/.test(r));
      if (esta) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    if (!quieto) window.scrollTo(0, 0);
  }
  window.addEventListener('hashchange', pintar);
  pintar();
})();
