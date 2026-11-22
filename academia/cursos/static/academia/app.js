/* Lo poco que la academia hace en el navegador: el reproductor retoma donde quedó el alumno y avisa su avance, las
   preguntas de ordenar se mueven con sus flechas y la página de una orden pendiente se actualiza sola.
   Todo lo demás (quién puede ver qué, corregir una prueba, confirmar un pago) ocurre en el servidor. */
(function () {
  'use strict';

  // ---- Reproductor
  var video = document.getElementById('video');
  if (video && video.dataset.avance) {
    var csrf = (document.cookie.match(/(?:^|; )csrftoken=([^;]+)/) || [])[1] || '', ultimo = 0, completada = false;
    var avisar = function (termino) {
      fetch(video.dataset.avance, {
        method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf },
        body: JSON.stringify({ segundos: Math.round(video.currentTime), completada: !!termino })
      }).then(function (r) { return r.ok ? r.json() : null; }).then(function (d) {
        var boton = document.getElementById('marcar');
        if (d && d.completada && boton) boton.textContent = 'Vista ✓ (desmarcar)';
      }).catch(function () { /* sin conexión: el próximo aviso lo corrige */ });
    };
    video.addEventListener('loadedmetadata', function () {
      var desde = +video.dataset.desde || 0;
      if (desde > 5 && desde < video.duration - 5) video.currentTime = desde;
    });
    video.addEventListener('timeupdate', function () {
      if (Math.abs(video.currentTime - ultimo) < 10) return;
      ultimo = video.currentTime;
      var termino = video.duration && video.currentTime > video.duration * .9 && !completada;
      if (termino) completada = true;
      avisar(termino);
    });
    video.addEventListener('pause', function () { avisar(false); });
    video.addEventListener('ended', function () { completada = true; avisar(true); });
  }

  // ---- Preguntas de ordenar: las flechas mueven la fila y el campo oculto lleva el orden que se entrega
  Array.prototype.forEach.call(document.querySelectorAll('.ordenable'), function (lista) {
    var campo = document.querySelector('input[name="' + lista.dataset.campo + '"]');
    var refrescar = function () {
      var filas = lista.querySelectorAll('li');
      campo.value = Array.prototype.map.call(filas, function (li) { return li.dataset.k; }).join(',');
      Array.prototype.forEach.call(filas, function (li, i) {
        li.querySelector('[data-mover="-1"]').disabled = i === 0;
        li.querySelector('[data-mover="1"]').disabled = i === filas.length - 1;
      });
    };
    lista.addEventListener('click', function (e) {
      var boton = e.target.closest('[data-mover]');
      if (!boton) return;
      var fila = boton.closest('li');
      if (boton.dataset.mover === '-1' && fila.previousElementSibling) lista.insertBefore(fila, fila.previousElementSibling);
      else if (boton.dataset.mover === '1' && fila.nextElementSibling) lista.insertBefore(fila.nextElementSibling, fila);
      refrescar();
    });
    refrescar();
  });

  // ---- Orden pendiente: la página se vuelve a pedir hasta que la pasarela confirme
  var pendiente = document.querySelector('[data-recargar]');
  if (pendiente) setTimeout(function () { location.reload(); }, (+pendiente.dataset.recargar || 6) * 1000);
})();
