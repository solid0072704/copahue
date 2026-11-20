/* Motor del video vertical para TikTok.

   La página trae las cuatro escenas ya compuestas; este archivo las mueve. Todo depende de un solo reloj: cuadro(t)
   deja la página tal como se ve a los t segundos, de modo que el mismo instante produce siempre la misma imagen y
   el video se captura cuadro a cuadro (capturar.cjs, con la misma interfaz que las láminas del curso).

   DATOS (lo escribe tiktok.py): marcas (segundo en que empieza a sonar cada frase del guion), rotulos (subtítulos
   con su tiempo), cine (las láminas reales del curso que se ven en la ventana, como secuencias de imágenes),
   titulo (lo que se escribe en el buscador), cifras del caso, total (segundos), fps, y zona, ancho y alto (para
   revisar que nada quede bajo la interfaz de TikTok). */
(function () {
  'use strict';
  var D = window.DATOS, M = D.marcas;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var lim = function (x) { return x < 0 ? 0 : x > 1 ? 1 : x; };
  var CURVA = {
    lin: function (p) { return p; },
    sale: function (p) { return 1 - Math.pow(1 - p, 3); },
    expo: function (p) { return p >= 1 ? 1 : 1 - Math.pow(2, -10 * p); },
    entraSale: function (p) { return p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2; },
    rebote: function (p) { var c = 1.7; return 1 + (c + 1) * Math.pow(p - 1, 3) + c * Math.pow(p - 1, 2); }
  };

  // ---- Pistas: cada una aporta al estado de un elemento (opacidad, posición, escala) o pinta algo por su cuenta
  var pistas = [], movidos = [], escenas = [];
  function pista(el, t0, dur, curva, fn) {
    if (el && !el._e) { el._e = {}; movidos.push(el); }
    pistas.push({ el: el, t0: t0, dur: dur, c: CURVA[curva] || CURVA.expo, fn: fn });
    return t0 + dur;
  }
  var libre = function (t0, dur, curva, fn) { return pista(null, t0, dur, curva, fn); };
  function entra(el, t0, o) {
    o = o || {};
    return pista(el, t0, o.dur || .55, o.curva || 'expo', function (p, e) {
      e.o *= lim(p * 1.6); e.y += (1 - p) * (o.dy === undefined ? 46 : o.dy); e.x += (1 - p) * (o.dx || 0);
      if (o.s) e.s *= o.s + (1 - o.s) * p;
    });
  }
  function sale(el, t0, o) {
    o = o || {};
    return pista(el, t0, o.dur || .28, 'entraSale', function (p, e) { e.o *= 1 - p; e.y += p * (o.dy === undefined ? -44 : o.dy); });
  }
  function variable(el, nombre, de, a, t0, dur, curva) {
    return libre(t0, dur, curva || 'expo', function (p) { el.style.setProperty(nombre, (de + (a - de) * p).toFixed(4)); });
  }
  function contar(el, t0, dur, de, a, formato, curva) {
    return libre(t0, dur, curva || 'sale', function (p) { el.textContent = formato(Math.round(de + (a - de) * p)); });
  }
  // Texto que sube palabra por palabra desde detrás de una máscara
  function palabras(el, t0, paso) {
    var nodos = [], w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null), n, k = 0, fin = t0;
    while ((n = w.nextNode())) nodos.push(n);
    nodos.forEach(function (nodo) {
      var frag = document.createDocumentFragment();
      nodo.nodeValue.split(/(\s+)/).forEach(function (trozo) {
        if (!trozo) return;
        if (/^\s+$/.test(trozo)) { frag.appendChild(document.createTextNode(' ')); return; }
        var m = document.createElement('span'), i = document.createElement('span');
        m.className = 'pl'; i.textContent = trozo; m.appendChild(i); frag.appendChild(m);
        fin = libre(t0 + (k++) * (paso || .07), .62, 'expo', function (p) { i.style.transform = 'translate3d(0,' + ((1 - p) * 114).toFixed(1) + '%,0)'; });
      });
      nodo.parentNode.replaceChild(frag, nodo);
    });
    return fin;
  }
  function escena(el, a, b) { escenas.push({ el: el, a: a, b: b }); pista(el, a, b - a, 'lin', function (p, e) { e.s *= 1 + .016 * p; }); }
  var miles = function (n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, '.'); };

  // ---- Fondo: una cinta de líneas finas que ondula despacio, como en las láminas del curso
  var lienzo, ctx;
  function fondo(t) {
    ctx.clearRect(0, 0, 1080, 1920);
    for (var j = 0; j < 30; j++) {
      var u = j / 29;
      ctx.beginPath();
      for (var x = -20; x <= 1100; x += 20) {
        var y = 1500 + u * 330 - x * .16 + Math.sin(x * .0042 + t * .32 + u * 2.3) * (58 + 34 * u) + Math.sin(x * .0093 - t * .21 + u * 5.1) * 22;
        if (x === -20) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = j % 3 ? 'rgba(242,234,219,' + (.035 + .05 * u).toFixed(3) + ')' : 'rgba(227,166,47,' + (.09 + .12 * u).toFixed(3) + ')';
      ctx.lineWidth = 1.3;
      ctx.stroke();
    }
  }

  // ---- Subtítulos: el que suena en este instante; entra en un parpadeo para que el cambio se note sin distraer
  var rot, rotP, rotActual = -2;
  function rotulos(t) {
    var i = -1;
    for (var k = 0; k < D.rotulos.length; k++) if (t >= D.rotulos[k].t0 && t < D.rotulos[k].t1) i = k;
    if (i !== rotActual) { rotActual = i; rotP.innerHTML = i < 0 ? '' : D.rotulos[i].html; }
    if (i < 0) return;
    var p = CURVA.sale(lim((t - D.rotulos[i].t0) / .16));
    rot.style.opacity = lim(p * 1.5).toFixed(3);
    rot.style.transform = 'translate3d(0,' + ((1 - p) * 16).toFixed(2) + 'px,0)';
  }

  // ---- Cine: las láminas del curso, una imagen por cuadro
  function cine(t) {
    var esperas = [];
    D.cine.forEach(function (c) {
      var img = c.img || (c.img = document.getElementById(c.id));
      var i = Math.max(0, Math.min(c.n - 1, Math.round((t - c.t0) * D.fps)));
      if (c.actual === i) return;
      c.actual = i;
      img.src = c.dir + '/' + ('0000' + (i + 1)).slice(-4) + '.jpg';
      esperas.push(img.decode());
    });
    return esperas.length ? Promise.all(esperas) : null;
  }

  // ---- La línea de tiempo --------------------------------------------------------------------------------------
  function montar() {
    var C = D.cifras;

    // 1 · El caso: el titular está desde el primer cuadro; cada cifra entra cuando la voz la nombra
    var caso = $('#caso'), h1 = $('h1', caso), tv = $('#t-ventas'), tu = $('#t-utilidad'), tc = $('#t-caja');
    escena(caso, 0, M.pregunta - .04);
    variable($('.k i', caso), '--k', 0, 1, 0, .6);
    pista(h1, 0, 1.1, 'expo', function (p, e) { e.s *= 1.04 - .04 * p; });
    [[tv, M.ventas + .14], [tu, M.utilidad + .04], [tc, M.caja + .04]].forEach(function (x) {
      entra(x[0], x[1], { dy: 44 });
      variable($('.fl', x[0]), '--r', 0, 1, x[1] + .05, .7);
    });
    contar($('.n', tv), M.ventas + .3, 1.2, 0, C.ventas_pct, function (n) { return '+' + n + ' %'; });
    contar($('.n', tu), M.utilidad + .16, 1.0, 0, C.utilidad, miles);
    // la caja parte en lo que había a comienzos de año y se vacía mientras la voz dice que no alcanza para los sueldos
    var cae = M.caja + .95;
    contar($('.n', tc), cae, 1.35, C.caja_antes, C.caja, miles, 'entraSale');
    variable(tc, '--w', 1, C.caja / C.caja_antes, cae, 1.35, 'entraSale');
    variable(tc, '--m', 0, 1, cae + .1, 1.0, 'entraSale');
    variable($('em', h1), '--r', 0, 1, cae + 1.2, .55);
    pista($('.val', tc), cae + 1.3, .5, 'lin', function (p, e) { e.s *= 1 + .05 * Math.sin(Math.PI * p); });
    sale(caso, M.pregunta - .34);

    // 2 · La idea: la pregunta, y la respuesta en dos palabras
    var idea = $('#idea'), preg = $('#preg'), docs = $$('.doc', idea);
    escena(idea, M.pregunta - .08, M.curso - .04);
    entra($('.k', idea), M.idea - .1, { dy: 0, dur: .4 });
    palabras(preg, M.pregunta - .06, .09);
    pista(preg, M.idea - .14, .6, 'entraSale', function (p, e) { e.y -= 324 * p; e.s *= 1 - .5 * p; });   // sube y queda de título
    entra($('.lu', idea), M.idea + .16, { dy: 56 });
    entra($('.ne', idea), M.idea + .84, { dy: 0, s: .3, curva: 'rebote', dur: .5 });
    entra($('.lc', idea), M.idea + 1.82, { dy: 56 });
    docs.forEach(function (d, i) { entra(d, M.estados + .04 + i * .13, { dy: 54 }); });
    var foco = M.mirar + .34;
    docs.forEach(function (d) {
      if (d.id === 'd-flujo') { variable(d, '--f', 0, 1, foco, .4); pista(d, foco, .5, 'rebote', function (p, e) { e.s *= 1 + .07 * p; }); }
      else pista(d, foco, .4, 'sale', function (p, e) { e.o *= 1 - .6 * p; });
    });
    sale(idea, M.curso - .32);

    // 3 · El curso: se ve una pregunta real del curso y lo que responde Claude; debajo, lo que te llevas
    var curso = $('#curso');
    escena(curso, M.curso - .06, M.busca - .04);
    entra($('.k', curso), M.curso - .04, { dy: 0, dur: .4 });
    palabras($('h2', curso), M.curso - .02, .06);
    entra($('#vent'), M.curso + .1, { dy: 80, s: .95, dur: .7 });
    entra($('#cine-b'), M.claude - .2, { dy: 0, dur: .4, curva: 'sale' });
    // cada ficha entra con la palabra que nombra, que no siempre abre la frase
    [['#c-metodo', M.metodo + .85], ['#c-prompts', M.prompts + .12], ['#c-excel', M.excel + .95]].forEach(function (x) {
      entra($(x[0]), x[1], { dy: 40, s: .9, curva: 'rebote', dur: .5 });
    });
    sale(curso, M.busca - .36);

    // 4 · Dónde está: el título se escribe en el buscador mientras la voz lo dice
    var cta = $('#cta'), q = $('#q');
    escena(cta, M.busca - .08, D.total + 1);
    entra($('.k', cta), M.busca - .04, { dy: 0, dur: .4 });
    palabras($('h2', cta), M.busca - .04, .08);
    entra($('#busca'), M.busca + .28, { dy: 44, dur: .55 });
    var escribe = M.titulo - .06;
    libre(escribe, 1.75, 'lin', function (p) { q.textContent = D.titulo.slice(0, Math.round(p * D.titulo.length)); });
    libre(0, D.total + 1, 'lin', function (p, e, t) { $('#busca').style.setProperty('--c', t > escribe + 1.75 && Math.floor((t - escribe) * 1.6) % 2 ? 0 : 1); });
    entra($('#video'), M.titulo + .5, { dy: 80, s: .95, dur: .7 });
    entra($('#meta'), M.titulo + 1.1, { dy: 30 });
  }

  // ---- Interfaz con capturar.cjs -----------------------------------------------------------------------------
  window.ajustar = function () {};
  // Lo que se ve en este instante y queda fuera de la zona segura: bajo las pestañas de arriba, sobre el texto de la
  // publicación y, de la mitad hacia abajo, a la izquierda de los botones
  window.revisar = function () {
    var Z = D.zona, fuera = [];
    $$('h1, h2, .k, .tile, .preg, .lock .pal, .lock .ne, .doc, .vent, .chip, .busca, .video, .meta, .rot p').forEach(function (el) {
      var r = el.getBoundingClientRect(), o = 1, n = el;
      if (!r.width || !r.height) return;
      for (; n && n !== document.body; n = n.parentElement) o *= parseFloat(getComputedStyle(n).opacity);
      if (o < .5) return;
      if (r.top < Z.arriba - 3 || r.bottom > D.alto - Z.abajo + 3 || r.left < 0 || r.right > D.ancho ||
          (r.bottom > Z.botones_desde && r.right > D.ancho - Z.derecha + 3)) {      // 3 px: lo que rebota al entrar
        fuera.push((el.id ? '#' + el.id : el.tagName.toLowerCase() + '.' + el.className) + ' [' + [r.left, r.top, r.right, r.bottom].map(Math.round).join(', ') + ']');
      }
    });
    return fuera;
  };
  window.iniciarLamina = function () {
    lienzo = $('#fondo'); ctx = lienzo.getContext('2d'); rot = $('#rot'); rotP = $('p', rot);
    if (D.guias) document.body.classList.add('guias');
    montar();
    return window.cuadro(0);
  };
  window.cuadro = function (t) {
    movidos.forEach(function (el) { var e = el._e; e.o = 1; e.x = 0; e.y = 0; e.s = 1; });
    pistas.forEach(function (k) { k.fn(k.c(lim((t - k.t0) / k.dur)), k.el && k.el._e, t); });
    escenas.forEach(function (s) { s.el.style.display = t >= s.a - 1e-6 && t < s.b ? '' : 'none'; });
    movidos.forEach(function (el) {
      var e = el._e;
      el.style.opacity = e.o >= .9995 ? '' : e.o.toFixed(4);
      el.style.transform = e.x || e.y || e.s !== 1 ? 'translate3d(' + e.x.toFixed(2) + 'px,' + e.y.toFixed(2) + 'px,0) scale(' + e.s.toFixed(5) + ')' : '';
    });
    fondo(t);
    rotulos(t);
    return cine(t);
  };
})();
