/* Motor de movimiento de las láminas.

   Cada lámina avanza por pasos (data-p: aparece en ese paso; data-foco: se destaca en esos pasos). Aquí se define
   cómo entra cada tipo de elemento y se expone una interfaz determinista para capturar el video cuadro a cuadro:

     prepararPaso(k)        -> segundos que dura la animación del paso k
     irPaso(k, t, tLamina)  -> deja la lámina tal como se ve t segundos después de empezar el paso k
                               (sin t: estado final). tLamina es el tiempo desde que empezó la lámina y mueve
                               lentamente la imagen de fondo de las portadas.

   No usa relojes ni azar: el mismo (k, t) produce siempre la misma imagen. */
(function () {
  'use strict';
  var $$ = function (sel, raiz) { return Array.prototype.slice.call((raiz || document).querySelectorAll(sel)); };
  var limitar = function (x) { return x < 0 ? 0 : x > 1 ? 1 : x; };
  var CURVA = {
    lin: function (p) { return p; },
    sale: function (p) { return 1 - Math.pow(1 - p, 3); },
    sale4: function (p) { return 1 - Math.pow(1 - p, 4); },
    expo: function (p) { return p >= 1 ? 1 : 1 - Math.pow(2, -10 * p); },
    entraSale: function (p) { return p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2; },
    rebote: function (p) { var c = 1.4; return 1 + (c + 1) * Math.pow(p - 1, 3) + c * Math.pow(p - 1, 2); }
  };

  // ---- Texto: cada palabra dentro de una máscara, para que suba línea por línea -------------------------
  var CON_MASCARA = 'h1, .seccion h2, .pregunta, .titulo-curso h2, .chips-tit, .frase-pie, .tarjeta .frase, .gracias';
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
        var s = document.createElement('span'), i = document.createElement('i');
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
  function tramo(t0, dur, curva, pintar) { T.push({ t0: t0, dur: dur, curva: CURVA[curva] || CURVA.sale, pintar: pintar }); return t0 + dur; }
  function duracion() { return T.reduce(function (m, x) { return Math.max(m, x.t0 + x.dur); }, 0); }
  function pintar(t) { T.forEach(function (x) { x.pintar(x.curva(limitar((t - x.t0) / x.dur)), t >= x.t0 + x.dur); }); }

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
      el.style.transform = 'translate3d(0,' + ((1 - p) * 40).toFixed(2) + 'px,0) scale(' + (.98 + .02 * p).toFixed(4) + ')';
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
      linea.forEach(function (palabra) {
        fin = Math.max(fin, tramo(t0 + i * (paso || .08), dur || .62, 'expo', function (p, f) {
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
  function escribir(pre, t0, dur) {   // el prompt se escribe de corrido, con cursor
    var nodos = [], w = document.createTreeWalker(pre, NodeFilter.SHOW_TEXT, null), n, total = 0;
    while ((n = w.nextNode())) { nodos.push({ n: n, texto: n.nodeValue, desde: total }); total += n.nodeValue.length; }
    var cursor = document.createElement('span'); cursor.className = 'cursor';
    return tramo(t0, dur, 'lin', function (p, fin) {
      var hasta = Math.round(total * p), ultimo = null;
      nodos.forEach(function (x) {
        var visibles = Math.max(0, Math.min(x.texto.length, hasta - x.desde));
        x.n.nodeValue = fin ? x.texto : x.texto.slice(0, visibles);
        if (visibles > 0) ultimo = x.n;
      });
      if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
      if (!fin && ultimo) ultimo.parentNode.insertBefore(cursor, ultimo.nextSibling);
    });
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
  function extras(el, t0) {           // detalles dentro de lo que entra: resaltados, semáforos y cifras
    var fin = t0;
    $$('mark', el).forEach(function (m, i) { fin = Math.max(fin, variable(m, '--r', 0, 1, t0 + .3 + i * .12, .5)); });
    $$('.semaforo', el).forEach(function (s, i) { fin = Math.max(fin, brotar(s, t0 + .28 + i * .05)); });
    return fin;
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
      $$('.con', el).forEach(function (c) { fin = Math.max(fin, aparecer(c, t0, .25, 0)); });
      $$('.barra', el).forEach(function (b, i) { fin = Math.max(fin, crecer(b, t0 + .08 + i * .06, .62)); });
      $$('.traza', el).forEach(function (l) { fin = Math.max(fin, trazar(l, t0, 1.25)); });
      $$('.punto', el).forEach(function (c, i, todos) { fin = Math.max(fin, brotar(c, t0 + .1 + 1.1 * i / Math.max(1, todos.length - 1), .26)); });
      $$('text, path, rect:not(.barra), line:not(.con)', el).forEach(function (x) { fin = Math.max(fin, aparecer(x, t0 + .34, .4, 0)); });
      return fin;
    }
    if (es('.puntos li, .panel li, .al, .rk, .fm')) {
      visible();
      fin = variable(el, '--r', 0, 1, t0, .55);
      Array.prototype.forEach.call(el.children, function (h, i) { fin = Math.max(fin, i === 0 ? desdeLado(h, t0 + .06, .4, -14) : aparecer(h, t0 + .1 + i * .07, .5, 16)); });
      if (q('.v')) fin = Math.max(fin, contar(q('.v'), t0 + .2, .8));
    } else if (es('.nt, .tarjeta, .dconc')) {
      visible();
      fin = variable(el, '--r', 0, 1, t0, .5);
      Array.prototype.forEach.call(el.children, function (h, i) { fin = Math.max(fin, desdeLado(h, t0 + .1 + i * .07, .5, -18)); });
      if (q('.gr')) fin = Math.max(fin, contar(q('.gr'), t0 + .15, .85));
      if (q('.frase')) fin = Math.max(fin, subirPalabras(q('.frase'), t0 + .12, .6, .07));
    } else if (es('.cifra')) {
      visible();
      fin = variable(el, '--r', 0, 1, t0, .6);
      Array.prototype.forEach.call(el.children, function (h, i) { fin = Math.max(fin, aparecer(h, t0 + .08 + i * .09, .5, 18)); });
      fin = Math.max(fin, contar(q('.val'), t0 + .16, 1.0));
    } else if (es('.panel, .mp, .bq, .op, .chip')) {
      visible();
      fin = variable(el, '--r', 0, 1, t0, .6);
      fin = Math.max(fin, hijos(el, t0 + .1, .07, 16));
      $$('li:not([data-p])', el).forEach(function (li, i) { fin = Math.max(fin, variable(li, '--r', 0, 1, t0 + .2 + i * .07, .5)); });
    } else if (es('tr')) {
      fin = aparecer(el, t0, .4, 10);
    } else if (es('.bl, .ms, .pc')) {
      fin = aparecer(el, t0, .4, 12);
      $$('p, li, h3, h4, .ind, .desl, .res, .mensaje, .tres > div', el).forEach(function (x, i) { fin = Math.max(fin, aparecer(x, t0 + .05 + i * .05, .4, 10)); });
    } else if (es('.frase-pie')) {
      visible(); fin = subirPalabras(el, t0, .6, .07);
    } else if (es('.gracias, .pregunta')) {
      visible(); fin = subirPalabras(el, t0, .7, .09);
    } else if (es('.remate, .docs, .dfila, .conecta .fuentes')) {
      visible(); fin = hijos(el, t0, .08, 16);
      $$('.fa b', el).forEach(function (b, i) { fin = Math.max(fin, contar(b, t0 + .15 + i * .06, .8)); });
    } else if (es('.mini-prompt')) {
      fin = aparecer(el, t0, .45, 14);
    } else {
      fin = aparecer(el, t0, .5, 20);
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

  function prepararPaso(k) {
    T = []; pasoActual = k;
    clases(k);
    var t0 = 0;
    if (k === 0) {                       // llega la lámina: imagen, antetítulo, título y pie
      var heroe = document.querySelector('.heroe');
      if (heroe) variable(heroe, '--h', 0, 1, 0, 1.1, 'sale');
      var kicker = document.querySelector('.kicker, .seccion .num, .titulo-curso .sobre');
      if (kicker) { variable(kicker, '--r', 0, 1, 0, .5); aparecer(kicker, 0, .4, 0); }
      var titulo = document.querySelector('.cab h1, .seccion h2, .titulo-curso h2, .chips-tit');
      if (titulo) t0 = subirPalabras(titulo, .08, .66, .09) - .34;
      $$('.seccion .baj, .titulo-curso .sub, .titulo-curso .quien').forEach(function (el, i) { if (!el.dataset.p) aparecer(el, t0 + i * .12, .55, 16); });
      $$('.seccion .it').forEach(function (it, i) { aparecer(it, t0 + .15 + i * .05, .4, 10); });
      $$('.prompt, .resp, .memo, .proyecto, .tabla-caja thead, .conecta .fle, .conecta .nodo, .metodo, .grafico > svg > line, .grafico > svg > text, .grafico > svg > rect').forEach(function (el, i) {
        if (el.closest('[data-p]')) return;
        var cuando = Math.max(.2, t0) + i * .03;
        if (el.matches('.prompt, .resp, .memo, .proyecto')) surgir(el, cuando); else aparecer(el, cuando, .5, 0);
      });
      var pre = document.querySelector('.prompt pre');
      if (pre) escribir(pre, Math.max(.45, t0 + .1), Math.min(2.6, .9 + pre.textContent.length / 520));
    }
    // lo que entra en este paso; al llegar la lámina también entra lo que está siempre a la vista
    var base = k === 0 ? Math.max(.22, t0) : 0, i = 0;
    var entran = $$(k === 0 ? '[data-p="0"], ' + SIEMPRE : '[data-p="' + k + '"]').filter(function (el) {
      if (el.dataset.p !== undefined) return +el.dataset.p === k;                    // lo de otros pasos entra en su paso
      return !el.closest('[data-p]');                                               // o con su contenedor, en el suyo
    });
    entran.forEach(function (el) {
      for (var a = el.parentElement; a; a = a.parentElement) if (entran.indexOf(a) >= 0) return;   // entra con su contenedor
      entrar(el, base + i * (el.matches('tr') ? .035 : .09)); i += 1;
    });
    // foco: lo que se destaca y lo que deja de destacarse en este paso
    $$('[data-foco]').forEach(function (el) {
      var antes = k > 0 && enFoco(el, k - 1), ahora = enFoco(el, k);
      if (antes !== ahora) variable(el, '--f', antes ? 1 : 0, ahora ? 1 : 0, k === 0 ? base + .3 : 0, .45, 'entraSale');
    });
    // en el método, el paso que se está nombrando se destaca y el anterior vuelve a su estado
    $$('.mp[data-p]').forEach(function (el) {
      if (el.dataset.foco) return;
      var p = +el.dataset.p;
      if (p === k) variable(el, '--f', 0, 1, .15, .4, 'entraSale'); else if (p === k - 1) variable(el, '--f', 1, 0, 0, .4, 'entraSale');
    });
    // en un prompt, lo que no se está explicando se atenúa
    $$('.prompt pre').forEach(function (pre) {
      var segs = $$('.seg', pre), antes = k > 0 && segs.some(function (s) { return s.dataset.foco && enFoco(s, k - 1); });
      var ahora = segs.some(function (s) { return s.dataset.foco && enFoco(s, k); });
      segs.forEach(function (s) {
        var a = antes && !(s.dataset.foco && enFoco(s, k - 1)) ? 1 : 0, b = ahora && !(s.dataset.foco && enFoco(s, k)) ? 1 : 0;
        if (a !== b) variable(s, '--d', a, b, 0, .45, 'entraSale'); else s.style.setProperty('--d', b);
      });
    });
    // la marca de «esto es lo que se está explicando» (bloques de una respuesta)
    $$('.bl[data-p]').forEach(function (el) {
      var p = +el.dataset.p;
      if (p === k) variable(el, '--a', 0, 1, .25, .45); else if (p === k - 1) variable(el, '--a', 1, 0, 0, .35); else el.style.removeProperty('--a');
    });
    return duracion();
  }

  function fondo(tLamina) {             // la imagen de las portadas llega asentándose y después se acerca muy despacio
    var img = document.querySelector('.heroe > i');
    if (!img) return;
    var t = tLamina || 0, llegada = 1 - CURVA.sale4(limitar(t / 1.9));
    img.style.setProperty('--z', (1.06 + .0045 * t + .07 * llegada).toFixed(5));
    img.style.setProperty('--x', (-1.1 * t).toFixed(2) + 'px');
  }

  function limpiar() {
    $$('[style]').forEach(function (el) {
      if (el.closest('.heroe') || el.classList.contains('heroe')) return;
      ['opacity', 'transform', 'stroke-dasharray', 'stroke-dashoffset', '--r', '--f', '--d', '--a', '--h'].forEach(function (p) { el.style.removeProperty(p); });
    });
  }

  window.prepararPaso = function (k) { limpiar(); return prepararPaso(k); };
  window.irPaso = function (k, t, tLamina) {
    if (pasoActual !== k) { limpiar(); prepararPaso(k); }
    pintar(t === undefined || t === null ? 1e9 : t);
    fondo(tLamina);
  };
  window.prepararLamina = function () { $$(CON_MASCARA).forEach(partir); };
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(window.prepararLamina);
})();
