/**
 * TPDV Academy · Puntos de entrada.
 *
 * doGet() sirve la página. Las demás funciones públicas (sin «_» al final) son las que el navegador
 * puede llamar con google.script.run; TODAS validan el dominio antes de hacer cualquier cosa.
 */

function doGet() {
  var usuario = usuarioActual_();
  if (!usuario.permitido) return paginaAccesoDenegado_(usuario.email);
  if (!idBaseDatos_()) {
    return paginaAviso_('La academia aún no está instalada',
      'Un administrador debe abrir el proyecto de Apps Script y ejecutar la función instalar().');
  }
  registrarIngreso_(usuario.email);
  var marca = marca_();
  var plantilla = HtmlService.createTemplateFromFile('Index');
  plantilla.marca = marca;
  plantilla.marcaJson = jsonSeguro_({ nombre: marca.nombre, bajada: marca.bajada, logo: marca.logo, logoColor: marca.logoColor });
  plantilla.datosJson = jsonSeguro_(estadoParaCliente_(usuario));
  return plantilla.evaluate()
    .setTitle(marca.nombre)
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

/** Inserta el contenido de otro archivo HTML del proyecto (se usa dentro de las plantillas). */
function incluir_(nombre) {
  return HtmlService.createHtmlOutputFromFile(nombre).getContent();
}

function jsonSeguro_(objeto) {
  return JSON.stringify(objeto)
    .replace(/</g, '\\u003c')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029');
}

function urlAplicacion_() {
  try { return ScriptApp.getService().getUrl() || ''; } catch (e) { return ''; }
}

function paginaAccesoDenegado_(email) {
  var plantilla = HtmlService.createTemplateFromFile('AccesoDenegado');
  plantilla.marca = marca_();
  plantilla.email = email;
  plantilla.dominio = DOMINIO_PERMITIDO;
  var url = urlAplicacion_();
  plantilla.urlCambiarCuenta = 'https://accounts.google.com/AccountChooser' + (url ? '?continue=' + encodeURIComponent(url) : '');
  return plantilla.evaluate()
    .setTitle('Acceso restringido · ' + plantilla.marca.nombre)
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

function paginaAviso_(titulo, mensaje) {
  var html = '<!doctype html><html><head><meta charset="utf-8"><style>' +
    'body{font-family:Arial,sans-serif;background:#F0F8FF;color:#1A1A1A;display:grid;place-items:center;min-height:100vh;margin:0}' +
    '.c{background:#fff;padding:32px;border-radius:16px;max-width:520px;box-shadow:0 4px 20px rgba(0,90,130,.12)}' +
    'h1{color:#005A82;font-size:22px}</style></head><body><div class="c"><h1></h1><p></p></div></body></html>';
  var salida = HtmlService.createHtmlOutput(html.replace('<h1></h1>', '<h1>' + escaparHtml_(titulo) + '</h1>')
    .replace('<p></p>', '<p>' + escaparHtml_(mensaje) + '</p>'));
  return salida.setTitle(titulo).addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

function escaparHtml_(texto) {
  return String(texto === undefined || texto === null ? '' : texto)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/** Todo lo que necesita la página del colaborador. */
function estadoParaCliente_(usuario) {
  var perfil = perfilUsuario_(usuario.email) || {};
  var estado = estadoUsuario_(usuario.email);
  var contenidos = {};
  var internos = typeof contenidoInterno_ === 'function' ? contenidoInterno_() : {};
  var puntaje = String(configNumero_('PUNTAJE_APROBACION', 0, 100));
  estado.modulos.forEach(function (m) {
    m.lecciones.forEach(function (l) {
      if (l.tipo === TIPOS_LECCION.TEXTO && internos[l.id]) {
        contenidos[l.id] = internos[l.id].replace(/\{\{PUNTAJE_APROBACION\}\}/g, puntaje);
      }
    });
  });
  var c = config_();
  return {
    usuario: {
      email: usuario.email,
      nombre: perfil.nombre || '',
      area: perfil.area || '',
      admin: usuario.admin
    },
    academia: {
      nombre: aTexto_(c['NOMBRE_ACADEMIA']) || 'TPDV Academy',
      bajada: aTexto_(c['BAJADA']),
      puntajeAprobacion: configNumero_('PUNTAJE_APROBACION', 0, 100),
      intentosMaximos: configNumero_('INTENTOS_MAXIMOS', 0, 50),
      ordenObligatorio: configSi_('ORDEN_OBLIGATORIO'),
      contacto: esCorreoPermitido_(c['CORREO_CONTACTO']) ? aTexto_(c['CORREO_CONTACTO']).toLowerCase() : '',
      version: VERSION_ACADEMIA
    },
    programa: estado,
    contenidos: contenidos,
    certificado: {
      disponible: estado.terminado,
      fecha: perfil.certificado || (estado.terminado ? new Date().toISOString() : '')
    }
  };
}

/* ------------------------------------------------------------------------------------------ */
/* API del colaborador                                                                        */
/* ------------------------------------------------------------------------------------------ */

function obtenerDatos() {
  return estadoParaCliente_(exigirUsuario_());
}

function marcarLeccion(idLeccion, completada) {
  var usuario = exigirUsuario_();
  registrarLeccion_(usuario.email, aTexto_(idLeccion), completada !== false);
  var datos = estadoParaCliente_(usuario);
  actualizarResumen_(usuario.email, datos.programa);
  return datos;
}

function guardarPerfil(datos) {
  var usuario = exigirUsuario_();
  datos = datos || {};
  guardarPerfil_(usuario.email, datos.nombre, datos.area);
  return estadoParaCliente_(usuario);
}

function obtenerEvaluacion(idModulo) {
  var usuario = exigirUsuario_();
  return evaluacionParaAlumno_(usuario.email, aTexto_(idModulo));
}

function enviarEvaluacion(idModulo, respuestas) {
  var usuario = exigirUsuario_();
  var resultado = registrarIntento_(usuario.email, aTexto_(idModulo), respuestas || {});
  var datos = estadoParaCliente_(usuario);
  actualizarResumen_(usuario.email, datos.programa);
  return { resultado: resultado, datos: datos };
}

/* ------------------------------------------------------------------------------------------ */
/* API de administración                                                                      */
/* ------------------------------------------------------------------------------------------ */

function obtenerPanelAdmin() {
  exigirAdmin_();
  return panelAdmin_();
}

function sincronizarContenido() {
  exigirAdmin_();
  var resumen = sincronizarDesdeDrive_();
  return { resumen: resumen, panel: panelAdmin_() };
}

function verificarPermisos() {
  exigirAdmin_();
  return verificarPermisosContenido_();
}

function registrarEvaluacionPractica(emailColaborador, idModulo, puntaje, comentario) {
  var admin = exigirAdmin_();
  var resultado = registrarPractica_(admin.email, emailColaborador, aTexto_(idModulo), puntaje, comentario);
  return { resultado: resultado, panel: panelAdmin_() };
}

function panelAdmin_() {
  var estructura = leerEstructura_();
  var filasProgreso = leerTabla_(HOJAS.PROGRESO);
  var filasResultados = leerTabla_(HOJAS.RESULTADOS);
  var opciones = { ordenObligatorio: configSi_('ORDEN_OBLIGATORIO') };
  var modulosPractica = estructura.modulos.filter(function (m) { return m.tipo === 'practica'; })
    .map(function (m) { return { id: m.id, titulo: m.titulo }; });

  var colaboradores = leerTabla_(HOJAS.USUARIOS).map(function (u) {
    var email = aTexto_(u['Email']).toLowerCase();
    var estado = calcularEstado_(
      estructura,
      leerProgresoUsuario_(email, filasProgreso).completadas,
      leerResultadosUsuario_(email, filasResultados),
      opciones
    );
    var pendientes = estado.modulos.filter(function (m) {
      return m.practica && !m.practica.aprobada && !m.bloqueado && m.leccionesCompletas &&
        (!m.evaluacion || m.evaluacion.aprobada);
    }).map(function (m) { return { id: m.id, titulo: m.titulo }; });
    var actual = estado.modulos.filter(function (m) { return !m.completo; })[0];
    return {
      email: email,
      nombre: aTexto_(u['Nombre']),
      area: aTexto_(u['Área']),
      primerIngreso: aFechaIso_(u['Primer ingreso']),
      ultimoIngreso: aFechaIso_(u['Último ingreso']),
      avance: estado.avance,
      modulosCompletos: estado.modulosCompletos,
      totalModulos: estado.totalModulos,
      moduloActual: actual ? actual.titulo : 'Programa completo',
      certificado: aFechaIso_(u['Certificado']) || (estado.terminado ? 'sí' : ''),
      practicasPendientes: pendientes,
      evaluaciones: estado.modulos.filter(function (m) { return m.evaluacion || m.practica; }).map(function (m) {
        return {
          modulo: m.titulo,
          mejor: m.evaluacion ? m.evaluacion.mejorPuntaje : (m.practica ? m.practica.puntaje : null),
          intentos: m.evaluacion ? m.evaluacion.intentos : (m.practica && m.practica.registrada ? 1 : 0),
          aprobada: m.evaluacion ? m.evaluacion.aprobada && (!m.practica || m.practica.aprobada) : m.practica.aprobada
        };
      })
    };
  }).filter(function (c) { return esCorreoPermitido_(c.email); });

  colaboradores.sort(function (a, b) { return (b.ultimoIngreso || '').localeCompare(a.ultimoIngreso || ''); });
  var total = colaboradores.length;
  return {
    resumen: {
      colaboradores: total,
      avancePromedio: total ? Math.round(colaboradores.reduce(function (s, c) { return s + c.avance; }, 0) / total) : 0,
      certificados: colaboradores.filter(function (c) { return c.certificado; }).length,
      practicasPendientes: colaboradores.reduce(function (s, c) { return s + c.practicasPendientes.length; }, 0)
    },
    colaboradores: colaboradores,
    modulos: estructura.modulos.map(function (m) {
      return { id: m.id, titulo: m.titulo, lecciones: m.lecciones.length, preguntas: m.preguntas, tipo: m.tipo };
    }),
    modulosPractica: modulosPractica,
    porClasificar: contarPendientesClasificar_(),
    urlBaseDatos: 'https://docs.google.com/spreadsheets/d/' + idBaseDatos_() + '/edit',
    urlCarpeta: 'https://drive.google.com/drive/folders/' + (aTexto_(config_()['CARPETA_CONTENIDO_ID']) || CARPETA_ACADEMIA_ID)
  };
}
