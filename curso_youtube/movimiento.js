/* Motor de movimiento de las láminas.

   Cada lámina avanza por pasos (data-p: aparece en ese paso; data-foco: se destaca en esos pasos). Este archivo
   define cómo se mueve todo y expone una interfaz determinista para capturar el video cuadro a cuadro:

     iniciarLamina(info)    -> prepara la lámina para reproducirla de corrido. info: duraciones (segundos de cada
                               paso), inicio (segundo del video en que empieza), entra / sale ('barrido', 'fundido'
                               o ''), modIni / modFin (segundos en que empieza y termina su capítulo).
     cuadro(t)              -> deja la lámina tal como se ve t segundos después de empezar.

   y, para revisar el diseño con imágenes fijas:

     prepararPaso(k)        -> segundos que dura la entrada del paso k
     irPaso(k, t, tLamina)  -> estado del paso k a los t segundos (sin t: estado final)

   Hay tres capas de movimiento: el fondo (cinta de líneas, halos y motas, que siguen el reloj del video y por eso
   no saltan al cambiar de lámina), la entrada de cada elemento repartida a lo largo de su paso (lo que se nombra
   primero entra primero; los prompts se escriben y las respuestas se van generando) y los bucles que mantienen
   viva la pantalla entre un paso y otro (luz que recorre lo que se está explicando, cursor, avance del capítulo).

   No usa relojes ni azar sin semilla: el mismo instante produce siempre la misma imagen. */
(function () {
  'use strict';
  var $$ = function (sel, raiz) { return Array.prototype.slice.call((raiz || document).querySelectorAll(sel)); };
  var $ = function (sel, raiz) { return (raiz || document).querySelector(sel); };
  var limitar = function (x) { return x < 0 ? 0 : x > 1 ? 1 : x; };
  var entre = function (x, a, b) { return x < a ? a : x > b ? b : x; };
  var CURVA = {
    lin: function (p) { return p; },
    sale: function (p) { return 1 - Math.pow(1 - p, 3); },
    sale4: function (p) { return 1 - Math.pow(1 - p, 4); },
    expo: function (p) { return p >= 1 ? 1 : 1 - Math.pow(2, -10 * p); },
    entraSale: function (p) { return p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2; },
    rebote: function (p) { var c = 1.4; return 1 + (c + 1) * Math.pow(p - 1, 3) + c * Math.pow(p - 1, 2); }
  };

  // ---- Texto: cada palabra dentro de una máscara, para que suba línea por línea -------------------------
  var CON_MASCARA = 'h1, .seccion h2, .pregunta, .titulo-curso h2, .chips-tit, .frase-pie, .tarjeta .frase, .gracias, ' +
                    '.puntos .txt b, .op > b, .chip b, .asesoria .as-t b';
  function partir(el) {
    if (el.dataset.partido) return;
    el.dataset.partido = '1';
    var nodos = [], w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null), n;
    while ((n = w.nextNode())) nodos.push(n);
    nodos.forEach(function (nodo) {
      var frag = document.createDocumentFragment();
      nodo.nodeValue.split(/(\s+)/).forEach(function (trozo) {
        if (!trozo) return;
        if (/^\s+$/.test(trozo)) { frag.appendChild(document.createTextNode(' ')); return; }
        var s = document.createElement('x-m'), i = document.createElement('x-p');     // etiquetas propias: no las alcanza «.chip span»
        s.className = 'pl'; i.textContent = trozo; s.appendChild(i); frag.appendChild(s);
      });
      nodo.parentNode.replaceChild(frag, nodo);
    });
  }
  function lineas(el) {           // palabras agrupadas por línea, según dónde quedaron al componer
    var grupos = [], ultimo = null;
    $$('.pl', el).forEach(function (pl) {
      var y = Math.round(pl.getBoundingClientRect().top);
      if (ultimo === null || Math.abs(y - ultimo) > 8) { grupos.push([]); ultimo = y; }
      grupos[grupos.length - 1].push(pl.firstChild);
    });
    return grupos;
  }

  // ---- Línea de tiempo del paso ---------------------------------------------------------------------
  var T = [], pasoActual = -1;
  var DUR = 0;                    // segundos que dura el paso en el video (0: no se sabe, imágenes fijas)
  function tramo(t0, dur, curva, pintar, tardio) { T.push({ t0: t0, dur: dur, curva: CURVA[curva] || CURVA.sale, pintar: pintar, tardio: tardio }); return t0 + dur; }
  function duracion() { return T.reduce(function (m, x) { return Math.max(m, x.t0 + x.dur); }, 0); }
  function pintar(t) {
    T.forEach(function (x) {
      if (x.tardio && t < x.t0) return;            // lo tardío no toca nada hasta que le llega su momento
      x.pintar(x.curva(limitar((t - x.t0) / x.dur)), t >= x.t0 + x.dur);
    });
  }
  // Cuándo arranca algo que entra «más tarde» dentro del paso: una fracción de lo que dura, entre un mínimo y un
  // máximo, y siempre con tiempo para terminar antes del paso siguiente.
  function luego(fraccion, minimo, maximo) {
    if (!DUR) return minimo;
    return Math.max(minimo, Math.min(maximo === undefined ? 1e9 : maximo, DUR * fraccion, DUR - 1.3));
  }

  // Animaciones elementales. Todas dejan el estilo en línea limpio al terminar, para que mande la hoja de estilos.
  // No todo es opaco por diseño (una banda tenue de un gráfico, por ejemplo): la entrada llega hasta la opacidad
  // propia del elemento y no la sobrepasa.
  function propia(el) {
    var previa = el.style.opacity;
    el.style.opacity = '';
    var o = parseFloat(getComputedStyle(el).opacity);
    el.style.opacity = previa;
    return isNaN(o) ? 1 : o;
  }
  function aparecer(el, t0, dur, dy, curva) {
    if (dy === undefined) dy = 22;
    var tope = propia(el);
    return tramo(t0, dur || .5, curva || 'sale', function (p, fin) {
      if (fin) { el.style.opacity = ''; el.style.transform = ''; return; }
      el.style.opacity = p * tope; el.style.transform = dy ? 'translate3d(0,' + ((1 - p) * dy).toFixed(2) + 'px,0)' : '';
    });
  }
  function desdeLado(el, t0, dur, dx) {
    var tope = propia(el);
    return tramo(t0, dur || .5, 'sale', function (p, fin) {
      if (fin) { el.style.opacity = ''; el.style.transform = ''; return; }
      el.style.opacity = p * tope; el.style.transform = 'translate3d(' + ((1 - p) * (dx || -18)).toFixed(2) + 'px,0,0)';
    });
  }
  function surgir(el, t0, dur) {      // superficies grandes: suben y se asientan, sin quedarse a medio fundir
    el.style.transformOrigin = '50% 0';
    var tope = propia(el);
    return tramo(t0, dur || .7, 'sale4', function (p, fin) {
      if (fin) { el.style.opacity = ''; el.style.transform = ''; return; }
      el.style.opacity = limitar(p * 1.6) * tope;
      el.style.transform = 'translate3d(0,' + ((1 - p) * 46).toFixed(2) + 'px,0) scale(' + (.972 + .028 * p).toFixed(4) + ')';
    });
  }
  function variable(el, nombre, de, a, t0, dur, curva) {
    return tramo(t0, dur, curva || 'expo', function (p, fin) {
      if (fin && a === 1 && nombre === '--r') { el.style.removeProperty(nombre); return; }
      el.style.setProperty(nombre, (de + (a - de) * p).toFixed(4));
    });
  }
  function subirPalabras(el, t0, dur, paso) {
    var fin = t0;
    lineas(el).forEach(function (linea, i) {
      linea.forEach(function (palabra, j) {
        fin = Math.max(fin, tramo(t0 + i * (paso || .08) + j * .018, dur || .62, 'expo', function (p, f) {
          palabra.style.transform = f ? '' : 'translate3d(0,' + ((1 - p) * 112).toFixed(1) + '%,0)';
        }));
      });
    });
    return fin;
  }
  function contar(el, t0, dur) {     // la primera cifra del elemento sube desde cero con su mismo formato
    var w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null), nodo, m = null;
    while ((nodo = w.nextNode())) { m = /\d[\d.]*(?:,\d+)?/.exec(nodo.nodeValue); if (m) break; }
    if (!m) return t0;
    var original = nodo.nodeValue, cifra = m[0], antes = original.slice(0, m.index), despues = original.slice(m.index + cifra.length);
    var dec = cifra.indexOf(',') >= 0 ? cifra.length - cifra.indexOf(',') - 1 : 0;
    var miles = /\d\.\d{3}/.test(cifra), valor = parseFloat(cifra.replace(/\./g, '').replace(',', '.'));
    var formato = function (v) {
      var s = v.toFixed(dec).split('.');
      if (miles) s[0] = s[0].replace(/\B(?=(\d{3})+(?!\d))/g, '.');
      return s.join(',');
    };
    return tramo(t0, dur || .9, 'sale4', function (p, fin) { nodo.nodeValue = fin ? original : antes + formato(valor * p) + despues; });
  }
  // El texto se va revelando (un prompt que se escribe, una respuesta que se genera) sin mover la composición:
  // lo que falta por aparecer ocupa su lugar, invisible.
  function revelar(el, t0, dur, conCursor) {
    var partes = [], w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null), n, total = 0;
    while ((n = w.nextNode())) { if (n.nodeValue.length) { partes.push({ n: n, texto: n.nodeValue, desde: total }); total += n.nodeValue.length; } }
    var cursor = null, montado = false;
    function montar() {
      partes.forEach(function (x) {
        var caja = document.createElement('x-rv'), a = document.createElement('x-ra'), b = document.createElement('x-ro');
        caja.className = 'rv'; b.className = 'rv-o'; b.textContent = x.texto;
        caja.appendChild(a); caja.appendChild(b); x.n.parentNode.replaceChild(caja, x.n); x.caja = caja; x.a = a; x.b = b;
      });
      if (conCursor) { cursor = document.createElement('x-c'); cursor.className = 'cursor'; }
      montado = true;
    }
    function desmontar() {
      if (!montado) return;
      partes.forEach(function (x) { if (x.caja.parentNode) x.caja.parentNode.replaceChild(x.n, x.caja); });
      if (cursor && cursor.parentNode) cursor.parentNode.removeChild(cursor);
      montado = false;
    }
    return tramo(t0, dur, 'lin', function (p, fin) {
      if (fin) { desmontar(); return; }
      if (!montado) montar();
      var hasta = Math.round(total * p), ultimo = null;
      partes.forEach(function (x) {
        var v = Math.max(0, Math.min(x.texto.length, hasta - x.desde));
        x.a.textContent = x.texto.slice(0, v); x.b.textContent = x.texto.slice(v);
        if (v > 0) ultimo = x;
      });
      if (cursor) {
        if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
        if (ultimo && p > 0) ultimo.caja.insertBefore(cursor, ultimo.b);
      }
    });
  }
  function dibujar(svg, t0, dur) {    // un icono de línea se dibuja trazo a trazo
    var fin = t0;
    $$('path, line, polyline, polygon, circle, rect, ellipse', svg).forEach(function (s, i) {
      var largo = 0;
      try { largo = s.getTotalLength(); } catch (e) { return; }
      if (!largo || getComputedStyle(s).fill !== 'none') return;
      fin = Math.max(fin, tramo(t0 + i * .06, dur || .6, 'entraSale', function (p, f) {
        if (f) { s.style.strokeDasharray = ''; s.style.strokeDashoffset = ''; return; }
        s.style.strokeDasharray = largo.toFixed(2); s.style.strokeDashoffset = (largo * (1 - p)).toFixed(2);
      }));
    });
    return fin;
  }
  function pulso(el, t0, cuanto, dur) {   // un toque: crece un poco y vuelve
    return tramo(t0, dur || .5, 'lin', function (p, fin) {
      el.style.transform = fin ? '' : 'scale(' + (1 + (cuanto || .12) * Math.sin(Math.PI * p)).toFixed(4) + ')';
    }, true);
  }
  function crecer(barra, t0, dur) {   // barras de los gráficos: crecen desde su base
    var dir = barra.getAttribute('data-dir') || 'arriba', eje = dir === 'derecha' ? 'X' : 'Y';
    barra.style.transformBox = 'fill-box';
    barra.style.transformOrigin = dir === 'derecha' ? 'left center' : dir === 'abajo' ? 'center top' : 'center bottom';
    return tramo(t0, dur || .6, 'expo', function (p, fin) { barra.style.transform = fin ? '' : 'scale' + eje + '(' + Math.max(.001, p).toFixed(4) + ')'; });
  }
  function trazar(linea, t0, dur) {   // líneas: se dibujan de principio a fin
    var largo = linea.getTotalLength();
    return tramo(t0, dur || 1.1, 'entraSale', function (p, fin) {
      if (fin) { linea.style.strokeDasharray = ''; linea.style.strokeDashoffset = ''; return; }
      linea.style.strokeDasharray = largo; linea.style.strokeDashoffset = (largo * (1 - p)).toFixed(2);
    });
  }
  function brotar(el, t0, dur) {
    el.style.transformBox = 'fill-box'; el.style.transformOrigin = 'center';
    var tope = propia(el);
    return tramo(t0, dur || .34, 'rebote', function (p, fin) {
      if (fin) { el.style.transform = ''; el.style.opacity = ''; return; }
      el.style.opacity = limitar(p * 3) * tope; el.style.transform = 'scale(' + Math.max(.001, p).toFixed(3) + ')';
    });
  }
  function hijos(el, t0, paso, dy) {  // los hijos directos entran escalonados
    var fin = t0;
    Array.prototype.forEach.call(el.children, function (h, i) { fin = Math.max(fin, aparecer(h, t0 + i * (paso || .06), .45, dy === undefined ? 14 : dy)); });
    return fin;
  }
  function extras(el, t0) {           // detalles dentro de lo que entra: resaltados, semáforos e iconos
    var fin = t0;
    $$('mark', el).forEach(function (m, i) { fin = Math.max(fin, variable(m, '--r', 0, 1, t0 + luego(.3, .4, 3) + i * .15, .6)); });
    $$('.semaforo', el).forEach(function (s, i) { fin = Math.max(fin, brotar(s, t0 + .28 + i * .07)); });
    $$('svg.ico', el).forEach(function (s, i) { fin = Math.max(fin, dibujar(s, t0 + .08 + i * .05, .62)); });
    return fin;
  }
  // El texto principal de un elemento: si quedó partido en palabras, sube palabra por palabra.
  function titular(el, t0) {
    if (!el) return t0;
    if (el.dataset.partido) return subirPalabras(el, t0, .6, .07);
    return aparecer(el, t0, .5, 18);
  }

  // ---- Cómo entra cada tipo de elemento --------------------------------------------------------------
  var SIEMPRE = '.puntos li, .panel, .cifra, .tabla tbody tr, .mp, .bq, .rk, .fm, .al, .op, .chip, .nt, .tarjeta, .bl, .ms, .dfila, .vs, ' +
                '.conecta .fuentes, .frase-pie, .sub-pie, .sobre-cifras, .mini-prompt, .docs, .pregunta, .remate, .gracias, .dconc, .pc, .lateral > *';
  function entrar(el, t0) {
    var es = function (sel) { return el.matches(sel); }, fin = t0, q = function (sel) { return el.querySelector(sel); };
    el.style.opacity = '';
    var visible = function () { tramo(t0, .001, 'lin', function (p, f) { el.style.opacity = f || p > 0 ? '' : 0; }); };
    if (el instanceof SVGElement) {
      visible();
      $$('.con', el).forEach(function (c) { fin = Math.max(fin, aparecer(c, t0, .3, 0)); });
      $$('.barra', el).forEach(function (b, i) { fin = Math.max(fin, crecer(b, t0 + .1 + i * .12, .9)); });
      var trazo = entre((DUR || 4) * .3, 1.3, 2.8);
      $$('.traza', el).forEach(function (l) { fin = Math.max(fin, trazar(l, t0 + .1, trazo)); });
      $$('.punto', el).forEach(function (c, i, todos) { fin = Math.max(fin, brotar(c, t0 + .15 + (trazo - .1) * i / Math.max(1, todos.length - 1), .3)); });
      $$('text, path, rect:not(.barra), line:not(.con)', el).forEach(function (x, i) { fin = Math.max(fin, aparecer(x, t0 + .45 + Math.min(i, 14) * .04, .45, 0)); });
      return fin;
    }
    if (es('.puntos li')) {
      visible();
      fin = variable(el, '--r', 0, 1, t0, .8);
      var marca = q('.marca'), b = q('.txt b'), sp = q('.txt span');
      if (marca) fin = Math.max(fin, brotar(marca, t0 + .05, .45));
      fin = Math.max(fin, titular(b, t0 + .14));
      if (sp) fin = Math.max(fin, aparecer(sp, t0 + luego(.36, .42, 4.5), .6, 16));
      if (q('.v')) fin = Math.max(fin, contar(q('.v'), t0 + .2, 1.1));
    } else if (es('.panel li, .al, .rk, .fm')) {
      visible();
      fin = variable(el, '--r', 0, 1, t0, .7);
      Array.prototype.forEach.call(el.children, function (h, i) { fin = Math.max(fin, i === 0 ? desdeLado(h, t0 + .06, .45, -26) : aparecer(h, t0 + .12 + i * .1, .55, 18)); });
      if (q('.v')) fin = Math.max(fin, contar(q('.v'), t0 + .2, 1.1));
    } else if (es('.nt, .tarjeta, .dconc')) {
      visible();
      fin = variable(el, '--r', 0, 1, t0, .6);
      Array.prototype.forEach.call(el.children, function (h, i) {
        var cuando = i === 0 ? .1 : h.matches('span, p') ? luego(.3, .25 + i * .08, 3.5) : .1 + i * .09;
        fin = Math.max(fin, desdeLado(h, t0 + cuando, .55, 34));
      });
      if (q('.gr')) { fin = Math.max(fin, contar(q('.gr'), t0 + .15, 1.3)); fin = Math.max(fin, pulso(q('.gr'), t0 + 1.5, .07, .5)); }
      if (q('.frase')) fin = Math.max(fin, subirPalabras(q('.frase'), t0 + .12, .6, .07));
    } else if (es('.cifra')) {
      visible();
      fin = variable(el, '--r', 0, 1, t0, .8);
      var val = q('.val'), sub = q('.sub'), antes = q('.antes'), eti = q('.eti');
      if (eti) fin = Math.max(fin, aparecer(eti, t0 + .08, .45, 12));
      if (val) {
        fin = Math.max(fin, aparecer(val, t0 + .16, .5, 26));
        fin = Math.max(fin, contar(val, t0 + .2, 1.6));
        fin = Math.max(fin, pulso(val, t0 + 1.85, .06, .5));
      }
      if (antes) fin = Math.max(fin, aparecer(antes, t0 + .5, .5, 12));
      if (sub) fin = Math.max(fin, aparecer(sub, t0 + luego(.34, .6, 4), .6, 14));
    } else if (es('.panel')) {
      visible();
      fin = variable(el, '--r', 0, 1, t0, .7);
      fin = Math.max(fin, hijos(el, t0 + .1, .08, 18));
      var paso = entre((DUR || 3) * .1, .1, .7);
      $$('li:not([data-p])', el).forEach(function (li, i) {
        fin = Math.max(fin, variable(li, '--r', 0, 1, t0 + .3 + i * paso, .6));
        fin = Math.max(fin, desdeLado(li, t0 + .3 + i * paso, .5, -22));
      });
    } else if (es('.mp, .bq, .op, .chip')) {
      visible();
      fin = variable(el, '--r', 0, 1, t0, .7);
      Array.prototype.forEach.call(el.children, function (h, i) {
        var tarde = h.matches('span, p') && i > 0;
        if (h.dataset.partido) fin = Math.max(fin, subirPalabras(h, t0 + .14 + i * .05, .6, .07));
        else fin = Math.max(fin, aparecer(h, t0 + (tarde ? luego(.3, .3 + i * .06, 3.5) : .1 + i * .07), .5, 18));
      });
    } else if (es('tr')) {
      fin = desdeLado(el, t0, .5, -30);
    } else if (es('.bl')) {             // un bloque de la respuesta de Claude: se va generando
      visible();
      fin = revelar(el, t0 + .1, entre((DUR || 3) * .5, .9, 7), true);
    } else if (es('.ms, .pc')) {
      fin = aparecer(el, t0, .45, 14);
      $$('p, li, h3, h4, .ind, .desl, .res, .mensaje, .tres > div', el).forEach(function (x, i) { fin = Math.max(fin, aparecer(x, t0 + .08 + i * .09, .45, 12)); });
    } else if (es('.frase-pie')) {
      visible(); fin = subirPalabras(el, t0, .6, .07);
    } else if (es('.gracias, .pregunta')) {
      visible(); fin = subirPalabras(el, t0, .7, .09);
    } else if (es('.remate, .docs, .dfila, .conecta .fuentes')) {
      visible(); fin = hijos(el, t0, .12, 22);
      $$('.fa b', el).forEach(function (b, i) { fin = Math.max(fin, contar(b, t0 + .15 + i * .08, 1.1)); });
    } else if (es('.asesoria')) {
      fin = surgir(el, t0, .75);
      $$('.as-t > *, .as-c > *', el).forEach(function (x, i) {
        fin = Math.max(fin, x.dataset.partido ? subirPalabras(x, t0 + .22 + i * .08, .6, .07) : aparecer(x, t0 + .2 + i * .1, .5, 14));
      });
    } else if (es('.mini-prompt')) {
      fin = surgir(el, t0, .6);
    } else {
      fin = aparecer(el, t0, .55, 24);
    }
    return Math.max(fin, extras(el, t0));
  }

  // ---- Estado de la lámina -----------------------------------------------------------------------------
  function clases(k) {
    document.body.dataset.paso = k;
    $$('[data-p]').forEach(function (el) { var p = +el.dataset.p; el.classList.toggle('visto', p <= k); el.classList.toggle('actual', p === k); });
    $$('[data-foco]').forEach(function (el) {
      var ps = el.dataset.foco.split(',').map(Number);
      el.classList.toggle('foco', ps.indexOf(k) >= 0); el.classList.toggle('pasado', Math.max.apply(null, ps) < k);
    });
  }
  function enFoco(el, k) { return el.dataset.foco.split(',').map(Number).indexOf(k) >= 0; }

  var activo = null;              // el elemento que se está explicando en este paso: por él pasa la luz
  var finEntrada = 0;             // segundo del paso en que termina de entrar lo principal
  var finEscritura = 0;           // segundo del paso en que termina de escribirse el prompt (0: no hay)
  var CON_LUZ = 'li, .panel, .cifra, .op, .chip, .bq, .rk, .fm, .al, .mp, .nt, .tarjeta, .bl, tr, .asesoria, .dfila, .ms, .mini-prompt';

  function prepararPaso(k) {
    T = []; pasoActual = k;
    clases(k);
    finEscritura = 0;
    var flecha = $('.puntero');
    if (flecha) flecha.style.opacity = 0;
    var t0 = 0;
    if (k === 0) {                       // llega la lámina: imagen, antetítulo, título y pie
      var heroe = $('.heroe');
      if (heroe) variable(heroe, '--h', 0, 1, 0, 1.2, 'sale');
      var kicker = $('.kicker, .seccion .num, .titulo-curso .sobre');
      if (kicker) { variable(kicker, '--r', 0, 1, 0, .6); aparecer(kicker, 0, .45, 0); }
      var titulo = $('.cab h1, .seccion h2, .titulo-curso h2, .chips-tit');
      if (titulo) t0 = subirPalabras(titulo, .1, .7, .1) - .36;
      $$('.seccion .baj, .titulo-curso .sub, .titulo-curso .quien').forEach(function (el, i) { if (!el.dataset.p) aparecer(el, t0 + .1 + i * .18, .6, 22); });
      $$('.seccion .it').forEach(function (it, i) { aparecer(it, t0 + .3 + i * .08, .45, 14); });
      $$('.prompt, .resp, .memo, .proyecto, .tabla-caja thead, .conecta .fle, .conecta .nodo, .metodo, .grafico > svg > line, .grafico > svg > text, .grafico > svg > rect').forEach(function (el, i) {
        if (el.closest('[data-p]')) return;
        var cuando = Math.max(.2, t0) + i * .03;
        if (DUR && DUR < 3.4) cuando = .12 + i * .03;
        if (el.matches('.prompt, .resp, .memo, .proyecto')) surgir(el, cuando, DUR && DUR < 3.4 ? .5 : .7); else aparecer(el, cuando, .5, 0);
      });
      var pre = $('.prompt pre');
      var corto = DUR && DUR < 3.4;      // no hay tiempo para escribirlo: llega pegado, tramo por tramo
      if (pre && corto) {
        var tramos = $$('.seg', pre), cada = entre((DUR - .95) / Math.max(1, tramos.length) * .5, .02, .08);
        tramos.forEach(function (s, i) { aparecer(s, .5 + i * cada, .3, 0); });
      }
      if (pre && !corto) {               // el prompt se escribe mientras se presenta, y se envía
        var desde = Math.max(.6, t0 + .25), largo = entre((DUR || 4) * .6, 1.0, 11);
        if (DUR && desde + largo > DUR - .3) largo = Math.max(.5, DUR - .3 - desde);
        finEscritura = revelar(pre, desde, largo, true);
        var enviar = $('.prompt .pb .env'), puntero = $('.puntero');
        if (enviar && puntero && DUR && finEscritura + 1.5 < DUR) {
          var r = enviar.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
          puntero.style.opacity = 0;
          tramo(finEscritura + .15, .7, 'entraSale', function (p) {
            puntero.style.opacity = limitar(p * 4);
            puntero.style.transform = 'translate3d(' + (cx - 240 * (1 - p)).toFixed(1) + 'px,' + (cy - 150 * (1 - p) * (1 - p)).toFixed(1) + 'px,0)';
          }, true);
          pulso(enviar, finEscritura + .85, .22, .4);
          tramo(finEscritura + .85, .2, 'lin', function (p) { puntero.style.transform = 'translate3d(' + cx.toFixed(1) + 'px,' + cy.toFixed(1) + 'px,0) scale(' + (1 - .16 * Math.sin(Math.PI * p)).toFixed(3) + ')'; }, true);
          tramo(finEscritura + 1.15, .3, 'lin', function (p) { puntero.style.opacity = 1 - p; }, true);
        }
      }
    }
    // lo que entra en este paso; al llegar la lámina también entra lo que está siempre a la vista
    var base = k === 0 ? Math.max(.25, t0) : 0;
    var entran = $$(k === 0 ? '[data-p="0"], ' + SIEMPRE : '[data-p="' + k + '"]').filter(function (el) {
      if (el.dataset.p !== undefined) return +el.dataset.p === k;                    // lo de otros pasos entra en su paso
      return !el.closest('[data-p]');                                               // o con su contenedor, en el suyo
    });
    var propios = entran.filter(function (el) {
      for (var a = el.parentElement; a; a = a.parentElement) if (entran.indexOf(a) >= 0) return false;   // entra con su contenedor
      return true;
    });
    // si entran varios a la vez se reparten por el paso: la narración los va nombrando en orden
    var filas = propios.filter(function (el) { return el.matches('tr'); }).length;
    var otros = propios.length - filas;
    var entreFilas = entre((DUR || 2) * .45 / Math.max(1, filas), .05, .16);
    var entreOtros = otros > 1 ? entre((DUR || 2) * .55 / otros, .1, 1.6) : .1;
    var cuandoFila = base, cuandoOtro = base;
    finEntrada = base;
    propios.forEach(function (el) {
      var fin;
      if (el.matches('tr')) { fin = entrar(el, cuandoFila); cuandoFila += entreFilas; }
      else { fin = entrar(el, cuandoOtro); cuandoOtro += entreOtros; }
      finEntrada = Math.max(finEntrada, Math.min(fin, cuandoOtro + 1.2, cuandoFila + 1.2));
    });
    // foco: lo que se destaca y lo que deja de destacarse en este paso
    $$('[data-foco]').forEach(function (el) {
      var antes = k > 0 && enFoco(el, k - 1), ahora = enFoco(el, k);
      if (antes !== ahora) variable(el, '--f', antes ? 1 : 0, ahora ? 1 : 0, k === 0 ? base + .3 : 0, .55, 'entraSale');
    });
    // en el método, el paso que se está nombrando se destaca y el anterior vuelve a su estado
    $$('.mp[data-p]').forEach(function (el) {
      if (el.dataset.foco) return;
      var p = +el.dataset.p;
      if (p === k) variable(el, '--f', 0, 1, .15, .45, 'entraSale'); else if (p === k - 1) variable(el, '--f', 1, 0, 0, .45, 'entraSale');
    });
    // en un prompt, lo que no se está explicando se atenúa
    $$('.prompt pre').forEach(function (pre) {
      var segs = $$('.seg', pre), antes = k > 0 && segs.some(function (s) { return s.dataset.foco && enFoco(s, k - 1); });
      var ahora = segs.some(function (s) { return s.dataset.foco && enFoco(s, k); });
      segs.forEach(function (s) {
        var a = antes && !(s.dataset.foco && enFoco(s, k - 1)) ? 1 : 0, b = ahora && !(s.dataset.foco && enFoco(s, k)) ? 1 : 0;
        if (a !== b) variable(s, '--d', a, b, 0, .5, 'entraSale'); else s.style.setProperty('--d', b);
      });
    });
    // la marca de «esto es lo que se está explicando» (bloques de una respuesta)
    $$('.bl[data-p]').forEach(function (el) {
      var p = +el.dataset.p;
      if (p === k) variable(el, '--a', 0, 1, .25, .5); else if (p === k - 1) variable(el, '--a', 1, 0, 0, .4); else el.style.removeProperty('--a');
    });
    // por dónde pasa la luz: lo que entró en este paso o, si nada entró, lo que quedó destacado
    activo = propios.filter(function (el) { return !(el instanceof SVGElement) && el.matches(CON_LUZ); })[0] || null;
    if (k === 0 && propios.length > 1) activo = null;          // al llegar la lámina entra todo: no hay un «esto»
    if (!activo && k > 0) activo = $('tr.foco, .mp.foco, .mp.actual, li.foco, .seg.foco');
    return duracion();
  }

  function fondo(tLamina) {             // la imagen de las portadas llega asentándose y después se acerca muy despacio
    var img = $('.heroe > i');
    if (!img) return;
    var t = tLamina || 0, llegada = 1 - CURVA.sale4(limitar(t / 1.9));
    img.style.setProperty('--z', (1.06 + .0045 * t + .07 * llegada).toFixed(5));
    img.style.setProperty('--x', (-1.1 * t).toFixed(2) + 'px');
  }

  function limpiar() {
    $$('[style]').forEach(function (el) {
      if (el.closest('.heroe, .ambiente, .pie, .capa-fija') || el.classList.contains('heroe') || el.classList.contains('lienzo')) return;
      ['opacity', 'transform', 'stroke-dasharray', 'stroke-dashoffset', '--r', '--f', '--d', '--a', '--h'].forEach(function (p) { el.style.removeProperty(p); });
    });
    $$('.cursor').forEach(function (c) { if (c.parentNode) c.parentNode.removeChild(c); });
  }

  // ---- Fondo vivo: halos, motas y la cinta de líneas, con el reloj del video ----------------------------
  var MOTAS = [], luz = null, telon = null, pincel = null, degradado = null, FASE = 0, LINEAS = 22;
  var LONA = { x: 384, y: 330 }, ALTO = 1080 - LONA.y;
  function azar(semilla) {              // generador con semilla: las motas son siempre las mismas
    var a = semilla;
    return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function prepararAmbiente() {
    var lienzo = $('.lienzo');
    if (!lienzo || $('.ambiente')) return;
    var amb = document.createElement('div'), lona = document.createElement('canvas'), al = azar(11);
    amb.className = 'ambiente';
    // el lienzo cubre solo la zona por donde pasa la cinta (abajo y a la derecha): pintar menos pixeles es más rápido
    lona.width = 1920 - LONA.x; lona.height = 1080 - LONA.y;
    lona.style.left = LONA.x + 'px'; lona.style.top = LONA.y + 'px'; lona.style.width = lona.width + 'px'; lona.style.height = lona.height + 'px';
    amb.appendChild(lona);
    pincel = lona.getContext('2d');
    pincel.translate(-LONA.x, -LONA.y);
    degradado = pincel.createLinearGradient(0, 0, 1920, 0);
    degradado.addColorStop(.2, 'rgba(227,166,47,0)'); degradado.addColorStop(.62, 'rgba(227,166,47,.5)'); degradado.addColorStop(1, 'rgba(242,217,160,1)');
    var fija0 = $('.ondas');
    if (fija0) { FASE = parseFloat(fija0.getAttribute('data-fase') || '0'); LINEAS = $$('path', fija0).length || 22; }
    for (var i = 0; i < 22; i++) {
      var radio = 1 + al() * 1.7, oro = al() < .4;
      MOTAS.push({ r: radio, oro: oro, x: LONA.x + 60 + al() * (1800 - LONA.x), y: al() * ALTO, v: 5 + al() * 11, f: al() * 6.283, a: .12 + al() * .26, vaiven: 10 + al() * 22 });
    }
    lienzo.insertBefore(amb, lienzo.firstChild);
    document.body.classList.add('vivo');
    var fija = document.createElement('div');
    fija.className = 'capa-fija';
    fija.innerHTML = '<div class="luz"></div><svg class="puntero" viewBox="0 0 24 24"><path d="M4 2.5l15.5 9.2-6.9 1.6-3.4 6.4z"/></svg><div class="telon"></div>';
    lienzo.appendChild(fija);
    luz = $('.luz'); telon = $('.telon');
  }
  function ambiente(tg) {
    if (!pincel) return;
    pincel.clearRect(LONA.x, LONA.y, 1920 - LONA.x, 1080 - LONA.y);
    // la cinta: la misma de la imagen fija en tg = 0, que ondula y respira con el tiempo
    pincel.globalAlpha = .24; pincel.strokeStyle = degradado; pincel.lineWidth = 1.15;
    for (var i = 0; i < LINEAS; i++) {
      var respira = 1 + .2 * Math.sin(.29 * tg + .33 * i);
      pincel.beginPath();
      for (var j = 0; j <= 64; j++) {
        var u = j / 64;
        var centro = 1018 - 380 * Math.pow(u, 1.9) + 12 * Math.sin(.17 * tg + 2.2 * u);
        var ancho = (i - (LINEAS - 1) / 2) * (2.2 + 19 * u) * respira;
        var y = centro + ancho * Math.sin(1.9 * Math.PI * u + FASE + i * .075 + .38 * tg);
        if (j) pincel.lineTo(1920 * u, y); else pincel.moveTo(0, y);
      }
      pincel.stroke();
    }
    // las motas: suben despacio, con un vaivén y un titilar leves
    MOTAS.forEach(function (m) {
      var sube = ((m.y - m.v * tg) % ALTO + ALTO) % ALTO, y = LONA.y + sube, x = m.x + m.vaiven * Math.sin(.21 * tg + m.f);
      var borde = Math.min(1, sube / 90, (ALTO - sube) / 90);          // se desvanecen al llegar arriba y al nacer abajo
      pincel.globalAlpha = m.a * (.55 + .45 * Math.sin(.9 * tg + m.f)) * borde;
      pincel.fillStyle = m.oro ? '#E3A62F' : '#DCE6F2';
      pincel.beginPath(); pincel.arc(x, y, m.r, 0, 6.2832); pincel.fill();
    });
  }

  // ---- Entre un paso y otro: luz sobre lo que se explica, cursor y avance del capítulo --------------------
  var cursorFijo = null;
  function bucles(tl, tg) {
    if (luz) {
      var desde = finEntrada + .5, visible = false;
      if (activo && tl > desde) {
        var fase = ((tl - desde) % 3.6) / 1.35;
        if (fase <= 1) {
          var r = activo.getBoundingClientRect(), ancho = Math.min(300, r.width * .45), e = CURVA.entraSale(fase);
          luz.style.width = ancho.toFixed(0) + 'px';
          luz.style.transform = 'translate3d(' + (r.left + (r.width - ancho) * e).toFixed(1) + 'px,' + (r.top - 1).toFixed(1) + 'px,0)';
          luz.style.opacity = Math.sin(Math.PI * fase).toFixed(3);
          visible = true;
        }
      }
      if (!visible) luz.style.opacity = 0;
    }
    // el cursor sigue parpadeando al final del prompt mientras se presenta
    var pre = $('.prompt pre');
    if (pre && pasoActual === 0 && finEscritura && tl >= finEscritura) {
      if (!cursorFijo) { cursorFijo = document.createElement('x-c'); cursorFijo.className = 'cursor fijo'; }
      if (cursorFijo.parentNode !== pre) pre.appendChild(cursorFijo);
      cursorFijo.style.opacity = ((tl - finEscritura) % 1.06) < .56 ? 1 : 0;
    } else if (cursorFijo && cursorFijo.parentNode) cursorFijo.parentNode.removeChild(cursorFijo);
    // el tramo del capítulo en el pie avanza con el video
    if (INFO && INFO.modFin > INFO.modIni) {
      var tramoPie = $('.avance i.actual');
      if (tramoPie) tramoPie.style.setProperty('--w', (limitar((tg - INFO.modIni) / (INFO.modFin - INFO.modIni)) * 100).toFixed(2) + '%');
    }
  }

  // ---- De una lámina a la siguiente: lo que hay se va, o un telón cruza la pantalla ----------------------
  var INFO = null, inicios = [], total = 0;
  function transicion(t) {
    if (!INFO) return;
    var contenido = $$('.lienzo > *').filter(function (el) { return !el.matches('.ambiente, .ondas, .pie, .capa-fija'); });
    var p = 0, cuadro = 1 / 30;
    if (INFO.sale === 'fundido') {
      p = CURVA.entraSale(limitar((t - (total - .34)) / (.34 - cuadro)));
      contenido.forEach(function (el) {
        if (el.classList.contains('heroe')) { if (p > 0) el.style.setProperty('--h', (1 - p).toFixed(4)); return; }
        el.style.opacity = p > 0 ? (1 - p).toFixed(4) : '';
        el.style.transform = p > 0 ? 'translate3d(0,' + (-22 * p).toFixed(2) + 'px,0)' : '';
      });
    }
    if (!telon) return;
    var x = null;
    if (INFO.entra === 'barrido' && t < .62) x = 100 * CURVA.entraSale(limitar(t / .6));                       // se va hacia la derecha
    if (INFO.sale === 'barrido' && t > total - .5) x = -100 + 100 * CURVA.entraSale(limitar((t - (total - .5)) / (.5 - cuadro)));   // entra desde la izquierda
    if (x === null || x >= 100) { telon.style.opacity = 0; }
    else { telon.style.opacity = 1; telon.style.transform = 'translate3d(' + x.toFixed(3) + '%,0,0)'; }
  }

  // ---- Interfaz --------------------------------------------------------------------------------------
  function prepararLamina() { $$(CON_MASCARA).forEach(partir); prepararAmbiente(); }
  window.prepararLamina = prepararLamina;
  window.prepararPaso = function (k) { DUR = INFO ? INFO.duraciones[k] : 0; limpiar(); return prepararPaso(k); };
  window.irPaso = function (k, t, tLamina) {
    if (pasoActual !== k) { DUR = 0; limpiar(); prepararPaso(k); }
    pintar(t === undefined || t === null ? 1e9 : t);
    fondo(tLamina);
    ambiente(0);
    if (luz) luz.style.opacity = 0;
    if (telon) telon.style.opacity = 0;
  };
  window.iniciarLamina = function (info) {
    INFO = info; inicios = []; total = 0; pasoActual = -1;
    document.body.classList.add('continuo');
    info.duraciones.forEach(function (d) { inicios.push(total); total += d; });
    prepararLamina();
  };
  window.cuadro = function (t) {
    var k = 0;
    while (k + 1 < inicios.length && t >= inicios[k + 1]) k++;
    while (pasoActual < k) {             // se avanza paso a paso, dejando terminado el anterior
      if (pasoActual >= 0) pintar(1e9);
      DUR = INFO.duraciones[pasoActual + 1];
      limpiar();
      prepararPaso(pasoActual + 1);
    }
    var tl = t - inicios[k], tg = (INFO.inicio || 0) + t;
    pintar(tl);
    fondo(t);
    ambiente(tg);
    bucles(tl, tg);
    transicion(t);
  };
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(prepararLamina);
})();
