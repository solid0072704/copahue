/**
 * Datos de prueba compartidos: un Drive simulado con todos los archivos del catálogo
 * y utilidades para recorrer el programa como lo haría un colaborador.
 */
'use strict';

const assert = require('node:assert/strict');
const { crearEntorno } = require('./gas_mock');

const RAIZ = '1hCDqvmusCl0nnC8TvKChiH_9ibKsR_b1';
const ALUMNA = 'nueva.asesora@tuproyectodevida.cl';
const ADMIN = 'admin.academia@tuproyectodevida.cl';

/** Árbol de Drive con todos los archivos del catálogo + casos especiales. */
function arbolDePrueba(catalogo) {
  const arbol = {
    carpetas: {
      [RAIZ]: { nombre: 'TPDV ACADEMY', hijos: ['carpVentas', 'carpMkt', 'carpEsc'], archivos: ['atajoCarpetaVideos', 'archivoNuevoRaiz'] },
      carpVentas: { nombre: 'Ventas', hijos: ['carpHubspot'], archivos: [] },
      carpHubspot: { nombre: 'Hubspot', hijos: [], archivos: [] },
      carpMkt: { nombre: 'Marketing', hijos: [], archivos: ['atajoAlDocMkt'] },
      carpEsc: { nombre: 'Escrituración', hijos: [], archivos: ['scriptSuelto'] },
      carpetaExternaVideos000: { nombre: 'Videos externos', hijos: [], archivos: ['archivoEnCarpetaExterna'] }
    },
    archivos: {
      atajoCarpetaVideos: { nombre: 'VIDEOS (acceso directo)', mime: 'application/vnd.google-apps.shortcut', destino: 'carpetaExternaVideos000', mimeDestino: 'application/vnd.google-apps.folder' },
      archivoNuevoRaiz: { nombre: 'Nuevo instructivo.pdf', mime: 'application/pdf' },
      atajoAlDocMkt: { nombre: 'Acceso a Desarrollo Pilares', mime: 'application/vnd.google-apps.shortcut', destino: '100JEbybm2MVA1miyABrK9bS0K25fd1U7a1OlxhJBVEM', mimeDestino: 'application/vnd.google-apps.document' },
      scriptSuelto: { nombre: 'Código', mime: 'application/vnd.google-apps.script' },
      archivoEnCarpetaExterna: { nombre: 'No debe aparecer.mp4', mime: 'video/mp4' },
      '1XWLUOPw-nXc8ps8VOdk_sc7glwxIk_dD': { nombre: 'horizontal_blanco.png', mime: 'image/png', bytes: [137, 80, 78, 71] },
      '1PpMp46Vcb0dZiZeGX9L1aglMz1APL_RM': { nombre: 'Horizontal_multicolor.png', mime: 'image/png', bytes: [137, 80, 78, 71] }
    }
  };
  const destinoPorModulo = { 'VENTAS-1': 'carpVentas', 'VENTAS-2': 'carpHubspot', 'MARKETING-1': 'carpMkt', 'MARKETING-2': 'carpMkt' };
  catalogo.lecciones.forEach((l) => {
    const id = l['ID archivo'];
    if (!id || arbol.archivos[id]) return;
    if (l['Tipo'] === 'Carpeta') {
      arbol.carpetas[id] = { nombre: l['Título'], hijos: [], archivos: [] };
      return;
    }
    arbol.archivos[id] = { nombre: l['Título'], mime: 'application/pdf' };
    // El doc de Marketing solo existe como acceso directo; el resto se reparte en las carpetas.
    if (id === '100JEbybm2MVA1miyABrK9bS0K25fd1U7a1OlxhJBVEM') return;
    arbol.carpetas[destinoPorModulo[l['ID módulo']] || 'carpEsc'].archivos.push(id);
  });
  // Un archivo excluido aparece en Drive: no debe volver a «Por clasificar».
  arbol.archivos['1PLKHyog8kpZGk5MaO6GSz1CnwwFYiucW'] = { nombre: 'Cuestionario para evaluador.docx', mime: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' };
  arbol.carpetas.carpEsc.archivos.push('1PLKHyog8kpZGk5MaO6GSz1CnwwFYiucW');
  // Un archivo del programa sin compartir con el dominio.
  arbol.archivos['1pBt806NPAcvkM7Hh0vJkh_DavV_UWgyy9p0mFjt-bsA'].acceso = 'PRIVATE';
  return arbol;
}

function entornoInstalado(ajustarArbol) {
  const base = crearEntorno({ dueno: ADMIN, usuario: ADMIN });
  const catalogo = base.contexto.catalogo_();
  const arbol = arbolDePrueba(catalogo);
  if (ajustarArbol) ajustarArbol(arbol);
  const env = crearEntorno({ dueno: ADMIN, usuario: ADMIN, arbol });
  env.ejecutar('instalar');
  env.catalogo = catalogo;
  return env;
}

function respuestasCorrectas(env, idModulo) {
  const r = {};
  env.catalogo.preguntas.filter(p => p['ID módulo'] === idModulo).forEach(p => { r[p['N°']] = p['Correcta']; });
  return r;
}

function completarModulo(env, idModulo) {
  const modulo = env.ejecutar('obtenerDatos').programa.modulos.find(m => m.id === idModulo);
  modulo.lecciones.forEach(l => env.ejecutar('marcarLeccion', l.id, true));
  if (modulo.evaluacion) {
    const r = env.ejecutar('enviarEvaluacion', idModulo, respuestasCorrectas(env, idModulo));
    assert.equal(r.resultado.aprobado, true, 'debería aprobar ' + idModulo);
  }
}

module.exports = { RAIZ, ALUMNA, ADMIN, arbolDePrueba, entornoInstalado, respuestasCorrectas, completarModulo };
