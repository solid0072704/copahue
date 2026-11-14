/* Interacciones mínimas: galería, orden de categoría, buscador interno, página de gracias y medición de clics. */
(function () {
  "use strict";
  var base = document.body.getAttribute("data-base") || "";

  function normalizar(t) {
    return (t || "").toString().toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
  }
  function precio(n) {
    return n ? "$" + Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".") : "";
  }
  function corto(n) {
    if (!n) return "";
    if (n >= 10000) return Math.floor(n / 1000) + " mil";
    if (n >= 1000) return (n / 1000).toFixed(1).replace(".0", "").replace(".", ",") + " mil";
    return String(n);
  }
  function escapar(t) {
    var d = document.createElement("div");
    d.textContent = t == null ? "" : String(t);
    return d.innerHTML;
  }

  // Galería de fotos
  var principal = document.getElementById("foto-principal");
  document.querySelectorAll("[data-foto]").forEach(function (b) {
    b.addEventListener("click", function () {
      principal.src = b.getAttribute("data-foto");
      document.querySelectorAll("[data-foto]").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
    });
  });

  // Orden de productos en categorías
  document.querySelectorAll("[data-ordenar]").forEach(function (sel) {
    var grilla = document.querySelector(sel.getAttribute("data-ordenar"));
    if (!grilla) return;
    sel.addEventListener("change", function () {
      var campo = sel.value;
      var items = Array.prototype.slice.call(grilla.children);
      items.sort(function (a, b) {
        var num = function (el, k) { return parseFloat(el.getAttribute("data-" + k)) || 0; };
        if (campo === "precio-asc") return num(a, "precio") - num(b, "precio");
        if (campo === "precio-desc") return num(b, "precio") - num(a, "precio");
        if (campo === "pedidos") return num(b, "pedidos") - num(a, "pedidos");
        return num(b, "score") - num(a, "score");
      });
      items.forEach(function (el) { grilla.appendChild(el); });
    });
  });

  // Índice de productos (buscador y página de gracias)
  var indice = null;
  function cargarIndice() {
    if (!indice) indice = fetch(base + "/buscar.json").then(function (r) { return r.json(); });
    return indice;
  }
  function tarjeta(p) {
    return '<article class="tarjeta"><a class="tarjeta-img" href="' + escapar(p.u) + '" tabindex="-1" aria-hidden="true">' +
      '<img src="' + escapar(p.i) + '" alt="' + escapar(p.n) + '" width="400" height="400" loading="lazy" referrerpolicy="no-referrer"></a>' +
      '<div class="tarjeta-cuerpo"><span class="tarjeta-nicho">' + escapar(p.c) + '</span>' +
      '<h3 class="tarjeta-nombre"><a href="' + escapar(p.u) + '">' + escapar(p.n) + '</a></h3>' +
      '<p class="tarjeta-datos">' + (p.v ? '<span class="val">★ ' + Math.round(p.v) + '%</span>' : "") + (p.o ? "<span>" + corto(p.o) + " pedidos</span>" : "") + "</p>" +
      '<p class="tarjeta-pie"><span class="precio">' + precio(p.p) + "</span></p></div></article>";
  }

  var campo = document.querySelector("[data-buscador]");
  if (campo) {
    var info = document.querySelector("[data-info]");
    var resultados = document.querySelector("[data-resultados]");
    var params = new URLSearchParams(location.search);
    campo.value = params.get("q") || "";
    var buscar = function () {
      var q = normalizar(campo.value).trim();
      if (!q) { info.textContent = "Escribe lo que necesitas."; resultados.innerHTML = ""; return; }
      cargarIndice().then(function (lista) {
        var palabras = q.split(/\s+/);
        var encontrados = lista.map(function (p) {
          var texto = normalizar(p.n + " " + p.c + " " + p.k);
          var puntos = 0;
          palabras.forEach(function (w) { if (texto.indexOf(w) >= 0) puntos += normalizar(p.n).indexOf(w) >= 0 ? 2 : 1; });
          return { p: p, puntos: puntos };
        }).filter(function (x) { return x.puntos >= palabras.length; })
          .sort(function (a, b) { return b.puntos - a.puntos; });
        info.textContent = encontrados.length ? encontrados.length + " resultados para «" + campo.value + "»" : "No encontramos resultados para «" + campo.value + "». Prueba con otra palabra o revisa las categorías.";
        resultados.innerHTML = encontrados.slice(0, 60).map(function (x) { return tarjeta(x.p); }).join("");
      });
    };
    var t;
    campo.addEventListener("input", function () { clearTimeout(t); t = setTimeout(buscar, 150); });
    buscar();
  }

  // Página de gracias (modo dropshipping): muestra el producto y su link de pago
  var gracias = document.querySelector("[data-gracias]");
  if (gracias) {
    var slug = new URLSearchParams(location.search).get("p");
    var pagado = new URLSearchParams(location.search).get("pago") === "ok";
    var caja = gracias.querySelector("[data-gracias-producto]");
    var wa = gracias.getAttribute("data-whatsapp");
    if (pagado) gracias.querySelector("[data-gracias-texto]").textContent = "¡Pago recibido! Te enviaremos el seguimiento por correo.";
    if (slug) cargarIndice().then(function (lista) {
      var p = lista.filter(function (x) { return x.s === slug; })[0];
      if (!p) return;
      var html = '<p><strong>' + escapar(p.n) + "</strong> · " + precio(p.p) + "</p>";
      if (p.pg && !pagado) html += '<p><a class="boton-cta" href="' + escapar(p.pg) + '">Pagar ahora con Mercado Pago</a></p>';
      if (wa) html += '<p><a class="boton-sec" href="https://wa.me/' + escapar(wa) + "?text=" + encodeURIComponent("Hola, acabo de hacer un pedido de " + p.n) + '" rel="nofollow noopener" target="_blank">Escríbenos por WhatsApp</a></p>';
      caja.innerHTML = html;
    });
  }

  // Medición de clics en ofertas (si hay Google Analytics)
  document.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest("[data-evento]");
    if (a && typeof window.gtag === "function") {
      window.gtag("event", a.getAttribute("data-evento"), { producto: a.getAttribute("data-producto") });
    }
  });
})();
