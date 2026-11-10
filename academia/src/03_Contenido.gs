/**
 * TPDV Academy · Estructura del programa (módulos y lecciones) y conexión con Drive.
 */

var CLAVE_CACHE_ESTRUCTURA = 'estructura_v1';

/** Tipos de lección que entiende la plataforma. */
var TIPOS_LECCION = {
  VIDEO: 'Video',
  PDF: 'PDF',
  DOCUMENTO: 'Documento',
  PRESENTACION: 'Presentación',
  PLANILLA: 'Planilla',
  FORMULARIO: 'Formulario',
  WORD: 'Word',
  POWERPOINT: 'PowerPoint',
  EXCEL: 'Excel',
  IMAGEN: 'Imagen',
  CARPETA: 'Carpeta',
  YOUTUBE: 'YouTube',
  ENLACE: 'Enlace',
  TEXTO: 'Texto',
  ARCHIVO: 'Archivo'
};

function tipoDesdeMime_(mime) {
  var m = String(mime || '').toLowerCase();
  if (m.indexOf('video/') === 0) return TIPOS_LECCION.VIDEO;
  if (m === 'application/pdf') return TIPOS_LECCION.PDF;
  if (m === 'application/vnd.google-apps.document') return TIPOS_LECCION.DOCUMENTO;
  if (m === 'application/vnd.google-apps.presentation') return TIPOS_LECCION.PRESENTACION;
  if (m === 'application/vnd.google-apps.spreadsheet') return TIPOS_LECCION.PLANILLA;
  if (m === 'application/vnd.google-apps.form') return TIPOS_LECCION.FORMULARIO;
  if (m === 'application/vnd.google-apps.folder') return TIPOS_LECCION.CARPETA;
  if (m.indexOf('wordprocessingml') !== -1 || m === 'application/msword') return TIPOS_LECCION.WORD;
  if (m.indexOf('presentationml') !== -1 || m === 'application/vnd.ms-powerpoint') return TIPOS_LECCION.POWERPOINT;
  if (m.indexOf('spreadsheetml') !== -1 || m === 'application/vnd.ms-excel') return TIPOS_LECCION.EXCEL;
  if (m.indexOf('image/') === 0) return TIPOS_LECCION.IMAGEN;
  return TIPOS_LECCION.ARCHIVO;
}

/** IDs de Drive válidos: letras, números, guion y guion bajo. Evita inyectar texto en las URLs. */
function idDriveValido_(id) {
  return /^[A-Za-z0-9_-]{10,}$/.test(String(id || ''));
}

function idYoutube_(url) {
  var m = String(url || '').match(/(?:youtu\.be\/|youtube\.com\/(?:watch\?(?:.*&)?v=|embed\/|shorts\/))([A-Za-z0-9_-]{6,})/);
  return m ? m[1] : '';
}

function urlSegura_(url) {
  var u = String(url || '').trim();
  return /^https:\/\//i.test(u) ? u : '';
}

/**
 * Calcula cómo se muestra una lección:
 *  - embed: URL para el visor dentro de la academia ('' si no se puede incrustar).
 *  - abrir: URL para abrir el material en una pestaña nueva.
 */
function urlsLeccion_(leccion) {
  var tipo = leccion.tipo;
  var id = idDriveValido_(leccion.idArchivo) ? leccion.idArchivo : '';
  var url = urlSegura_(leccion.url);

  if (tipo === TIPOS_LECCION.TEXTO) return { embed: '', abrir: url };
  if (tipo === TIPOS_LECCION.YOUTUBE || idYoutube_(url)) {
    var yt = idYoutube_(url);
    return yt ? { embed: 'https://www.youtube.com/embed/' + yt + '?rel=0', abrir: 'https://www.youtube.com/watch?v=' + yt }
      : { embed: '', abrir: url };
  }
  if (!id) return { embed: '', abrir: url };

  switch (tipo) {
    case TIPOS_LECCION.DOCUMENTO:
      return { embed: 'https://docs.google.com/document/d/' + id + '/preview', abrir: 'https://docs.google.com/document/d/' + id + '/edit' };
    case TIPOS_LECCION.PRESENTACION:
      return { embed: 'https://docs.google.com/presentation/d/' + id + '/embed?start=false&loop=false', abrir: 'https://docs.google.com/presentation/d/' + id + '/edit' };
    case TIPOS_LECCION.PLANILLA:
      return { embed: 'https://docs.google.com/spreadsheets/d/' + id + '/preview', abrir: 'https://docs.google.com/spreadsheets/d/' + id + '/edit' };
    case TIPOS_LECCION.FORMULARIO:
      return { embed: 'https://docs.google.com/forms/d/' + id + '/viewform?embedded=true', abrir: 'https://docs.google.com/forms/d/' + id + '/viewform' };
    case TIPOS_LECCION.CARPETA:
      return { embed: 'https://drive.google.com/embeddedfolderview?id=' + id + '#list', abrir: 'https://drive.google.com/drive/folders/' + id };
    case TIPOS_LECCION.ARCHIVO:
      // Formatos que Drive no previsualiza (por ejemplo Keynote): solo se abre o descarga.
      return { embed: '', abrir: 'https://drive.google.com/file/d/' + id + '/view' };
    default:
      // Video, PDF, Word, PowerPoint, Excel, imágenes: visor de Drive.
      return { embed: 'https://drive.google.com/file/d/' + id + '/preview', abrir: 'https://drive.google.com/file/d/' + id + '/view' };
  }
}

/**
 * Estructura visible del programa, ordenada. Se guarda en caché 10 minutos.
 * { modulos: [{id, orden, semana, pilar, titulo, descripcion, objetivos[], tipo, lecciones[], preguntas}] }
 */
function leerEstructura_() {
  var cache = CacheService.getScriptCache();
  var guardada = cache.get(CLAVE_CACHE_ESTRUCTURA);
  if (guardada) {
    try { return JSON.parse(guardada); } catch (e) { /* se recalcula */ }
  }
  var estructura = construirEstructura_(leerTabla_(HOJAS.MODULOS), leerTabla_(HOJAS.LECCIONES), leerTabla_(HOJAS.PREGUNTAS));
  var json = JSON.stringify(estructura);
  if (json.length < 95000) cache.put(CLAVE_CACHE_ESTRUCTURA, json, 600);
  return estructura;
}

function invalidarEstructura_() {
  CacheService.getScriptCache().remove(CLAVE_CACHE_ESTRUCTURA);
}

/** Convierte las filas de las hojas en la estructura del programa (función pura, se prueba sin Google). */
function construirEstructura_(filasModulos, filasLecciones, filasPreguntas) {
  var preguntasPorModulo = {};
  filasPreguntas.forEach(function (p) {
    var id = aTexto_(p['ID módulo']);
    if (id && aTexto_(p['Pregunta'])) preguntasPorModulo[id] = (preguntasPorModulo[id] || 0) + 1;
  });

  var modulos = filasModulos
    .filter(function (m) { return aTexto_(m['ID']) && esVisible_(m['Visible']); })
    .map(function (m) {
      var id = aTexto_(m['ID']);
      return {
        id: id,
        orden: Number(m['Orden']) || 999,
        semana: aTexto_(m['Semana']),
        pilar: aTexto_(m['Pilar']),
        titulo: aTexto_(m['Título']) || id,
        descripcion: aTexto_(m['Descripción']),
        objetivos: aTexto_(m['Objetivos']).split(/\n|•|;/).map(function (t) { return t.trim(); }).filter(String),
        tipo: aTexto_(m['Tipo']).toLowerCase() === 'práctica' || aTexto_(m['Tipo']).toLowerCase() === 'practica' ? 'practica' : 'contenido',
        lecciones: [],
        preguntas: preguntasPorModulo[id] || 0
      };
    });
  modulos.sort(function (a, b) { return a.orden - b.orden || a.titulo.localeCompare(b.titulo); });

  var porId = {};
  modulos.forEach(function (m) { porId[m.id] = m; });

  filasLecciones.forEach(function (l) {
    var modulo = porId[aTexto_(l['ID módulo'])];
    if (!modulo || !aTexto_(l['ID']) || !esVisible_(l['Visible'])) return;
    var leccion = {
      id: aTexto_(l['ID']),
      orden: Number(l['Orden']) || 999,
      seccion: aTexto_(l['Sección']),
      titulo: aTexto_(l['Título']) || 'Lección',
      tipo: normalizarTipo_(l['Tipo']),
      idArchivo: aTexto_(l['ID archivo']),
      url: aTexto_(l['URL']),
      descripcion: aTexto_(l['Descripción']),
      duracion: Number(l['Duración (min)']) || 0,
      obligatoria: aTexto_(l['Obligatoria']) === '' ? true : esSi_(l['Obligatoria'])
    };
    var urls = urlsLeccion_(leccion);
    leccion.embed = urls.embed;
    leccion.abrir = urls.abrir;
    modulo.lecciones.push(leccion);
  });

  modulos.forEach(function (m) {
    m.lecciones.sort(function (a, b) { return a.orden - b.orden || a.titulo.localeCompare(b.titulo); });
  });
  return { modulos: modulos };
}

function esVisible_(valor) {
  return aTexto_(valor) === '' ? true : esSi_(valor);
}

function normalizarTipo_(valor) {
  var t = aTexto_(valor).toLowerCase();
  var claves = Object.keys(TIPOS_LECCION);
  for (var i = 0; i < claves.length; i++) {
    var etiqueta = TIPOS_LECCION[claves[i]];
    if (etiqueta.toLowerCase() === t || claves[i].toLowerCase() === t) return etiqueta;
  }
  if (t === 'presentacion') return TIPOS_LECCION.PRESENTACION;
  return TIPOS_LECCION.ARCHIVO;
}

/* ------------------------------------------------------------------------------------------ */
/* Sincronización con Drive                                                                   */
/* ------------------------------------------------------------------------------------------ */

/**
 * Recorre la carpeta de la academia (subcarpetas reales, no accesos directos a otras carpetas) y
 * deja en «Por clasificar» todo archivo que aún no forme parte del programa.
 * Luego incorpora a «Lecciones» las filas de «Por clasificar» que tengan módulo asignado e Incluir = SI.
 */
function sincronizarDesdeDrive_() {
  var carpetaId = aTexto_(config_()['CARPETA_CONTENIDO_ID']) || CARPETA_ACADEMIA_ID;
  var encontrados = recorrerCarpeta_(DriveApp.getFolderById(carpetaId), '', 0, {}, []);

  return conBloqueo_(function () {
    var lecciones = leerTabla_(HOJAS.LECCIONES);
    var pendientes = leerTabla_(HOJAS.POR_CLASIFICAR);
    var conocidos = {};
    lecciones.forEach(function (l) { if (aTexto_(l['ID archivo'])) conocidos[aTexto_(l['ID archivo'])] = true; });
    pendientes.forEach(function (p) { if (aTexto_(p['ID archivo'])) conocidos[aTexto_(p['ID archivo'])] = true; });
    var idDb = idBaseDatos_();
    if (idDb) conocidos[idDb] = true;

    var ahora = new Date();
    var nuevos = encontrados.filter(function (a) { return !conocidos[a.id]; }).map(function (a) {
      return {
        'ID archivo': a.id,
        'Título': a.titulo,
        'Tipo': a.tipo,
        'Carpeta': a.carpeta,
        'URL': a.url,
        'Detectado': ahora,
        'Asignar a módulo (ID)': '',
        'Incluir (SI/NO)': '',
        'Nota': ''
      };
    });
    anexarFilas_(HOJAS.POR_CLASIFICAR, nuevos);

    var incorporados = incorporarClasificados_(lecciones);
    invalidarEstructura_();
    return {
      revisados: encontrados.length,
      nuevos: nuevos.length,
      incorporados: incorporados,
      pendientes: contarPendientesClasificar_()
    };
  });
}

function recorrerCarpeta_(carpeta, ruta, profundidad, visitadas, salida) {
  if (profundidad > 6 || salida.length > 3000) return salida;
  var idCarpeta = carpeta.getId();
  if (visitadas[idCarpeta]) return salida;
  visitadas[idCarpeta] = true;
  var rutaActual = ruta ? ruta + ' / ' + carpeta.getName() : carpeta.getName();

  var archivos = carpeta.getFiles();
  while (archivos.hasNext()) {
    var archivo = archivos.next();
    if (archivo.isTrashed && archivo.isTrashed()) continue;
    var mime = archivo.getMimeType();
    var id = archivo.getId();
    if (mime === 'application/vnd.google-apps.shortcut') {
      var destino = archivo.getTargetId && archivo.getTargetId();
      var mimeDestino = archivo.getTargetMimeType && archivo.getTargetMimeType();
      if (!destino || mimeDestino === 'application/vnd.google-apps.folder') continue;
      id = destino;
      mime = mimeDestino;
    }
    if (mime === 'application/vnd.google-apps.script') continue;
    salida.push({
      id: id,
      titulo: archivo.getName(),
      tipo: tipoDesdeMime_(mime),
      carpeta: rutaActual,
      url: 'https://drive.google.com/open?id=' + id
    });
  }
  var subcarpetas = carpeta.getFolders();
  while (subcarpetas.hasNext()) {
    recorrerCarpeta_(subcarpetas.next(), rutaActual, profundidad + 1, visitadas, salida);
  }
  return salida;
}

/** Mueve a «Lecciones» las filas clasificadas (módulo válido + Incluir = SI). Debe llamarse con bloqueo. */
function incorporarClasificados_(leccionesActuales) {
  var modulos = {};
  leerTabla_(HOJAS.MODULOS).forEach(function (m) { modulos[aTexto_(m['ID'])] = true; });
  var maxOrden = {};
  leccionesActuales.forEach(function (l) {
    var idm = aTexto_(l['ID módulo']);
    maxOrden[idm] = Math.max(maxOrden[idm] || 0, Number(l['Orden']) || 0);
  });

  var nuevas = [];
  leerTabla_(HOJAS.POR_CLASIFICAR).forEach(function (p) {
    var idModulo = aTexto_(p['Asignar a módulo (ID)']);
    if (!idModulo || !esSi_(p['Incluir (SI/NO)']) || aTexto_(p['Nota']).indexOf('Incorporado') === 0) return;
    if (!modulos[idModulo]) {
      actualizarFila_(HOJAS.POR_CLASIFICAR, p._fila, { 'Nota': 'Revisar: el módulo «' + idModulo + '» no existe.' });
      return;
    }
    maxOrden[idModulo] = (maxOrden[idModulo] || 0) + 1;
    nuevas.push({
      'ID': 'D-' + aTexto_(p['ID archivo']),
      'ID módulo': idModulo,
      'Orden': maxOrden[idModulo],
      'Sección': '',
      'Título': aTexto_(p['Título']).replace(/\.(pdf|docx?|pptx?|xlsx?|mp4|mov)$/i, ''),
      'Tipo': aTexto_(p['Tipo']) || TIPOS_LECCION.ARCHIVO,
      'ID archivo': aTexto_(p['ID archivo']),
      'URL': '',
      'Descripción': '',
      'Duración (min)': '',
      'Obligatoria': 'SI',
      'Visible': 'SI',
      'Origen': 'Drive'
    });
    actualizarFila_(HOJAS.POR_CLASIFICAR, p._fila, { 'Nota': 'Incorporado a ' + idModulo + ' el ' + Utilities.formatDate(new Date(), 'America/Santiago', 'dd-MM-yyyy') });
  });
  anexarFilas_(HOJAS.LECCIONES, nuevas);
  return nuevas.length;
}

function contarPendientesClasificar_() {
  return leerTabla_(HOJAS.POR_CLASIFICAR).filter(function (p) {
    return aTexto_(p['Incluir (SI/NO)']) === '';
  }).length;
}

/* ------------------------------------------------------------------------------------------ */
/* Verificación de permisos                                                                   */
/* ------------------------------------------------------------------------------------------ */

/**
 * Revisa cada archivo del programa: si la cuenta dueña del script no lo puede abrir, o si no está
 * compartido con el dominio, el colaborador nuevo verá el visor de Drive en blanco o «Solicitar acceso».
 */
function verificarPermisosContenido_() {
  var inicio = Date.now();
  var problemas = [];
  var revisados = 0;
  var accesosOk = {
    ANYONE: true, ANYONE_WITH_LINK: true, DOMAIN: true, DOMAIN_WITH_LINK: true
  };
  leerEstructura_().modulos.forEach(function (modulo) {
    modulo.lecciones.forEach(function (leccion) {
      if (!idDriveValido_(leccion.idArchivo) || Date.now() - inicio > 240000) return;
      revisados++;
      var base = { modulo: modulo.titulo, leccion: leccion.titulo, idArchivo: leccion.idArchivo, abrir: leccion.abrir };
      try {
        var archivo = leccion.tipo === TIPOS_LECCION.CARPETA ? DriveApp.getFolderById(leccion.idArchivo) : DriveApp.getFileById(leccion.idArchivo);
        var acceso = String(archivo.getSharingAccess());
        if (!accesosOk[acceso]) {
          var dueno = '';
          try { dueno = archivo.getOwner() ? archivo.getOwner().getEmail() : ''; } catch (e) { dueno = ''; }
          problemas.push(Object.assign(base, {
            problema: 'No está compartido con ' + DOMINIO_PERMITIDO + '. Pide a ' + (dueno || 'su dueño') +
              ' que lo comparta como «Cualquier persona de ' + DOMINIO_PERMITIDO + ' con el enlace: lector».'
          }));
        }
      } catch (e) {
        problemas.push(Object.assign(base, {
          problema: 'La cuenta dueña de la academia no puede abrir este archivo (solo se compartió un acceso directo o no hay permiso).'
        }));
      }
    });
  });
  return { revisados: revisados, problemas: problemas };
}

/* ------------------------------------------------------------------------------------------ */
/* Marca                                                                                      */
/* ------------------------------------------------------------------------------------------ */

function colorValido_(valor, porDefecto) {
  var v = aTexto_(valor);
  return /^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(v) ? v : porDefecto;
}

function fuenteValida_(valor, porDefecto) {
  var v = aTexto_(valor);
  return /^[A-Za-z0-9 ]{2,40}$/.test(v) ? v : porDefecto;
}

function valorDefecto_(clave) {
  return CONFIG_DEFECTO.filter(function (f) { return f[0] === clave; })[0][1];
}

/** Kit de marca listo para usar en la página (valores validados, sin riesgo de inyección CSS). */
function marca_() {
  var c = config_();
  var fuenteTitulos = fuenteValida_(c['FUENTE_TITULOS'], valorDefecto_('FUENTE_TITULOS'));
  var fuenteTexto = fuenteValida_(c['FUENTE_TEXTO'], valorDefecto_('FUENTE_TEXTO'));
  var familias = [fuenteTitulos, fuenteTexto].filter(function (f, i, arr) { return arr.indexOf(f) === i; });
  return {
    nombre: aTexto_(c['NOMBRE_ACADEMIA']) || 'TPDV Academy',
    bajada: aTexto_(c['BAJADA']),
    primario: colorValido_(c['COLOR_PRIMARIO'], valorDefecto_('COLOR_PRIMARIO')),
    secundario: colorValido_(c['COLOR_SECUNDARIO'], valorDefecto_('COLOR_SECUNDARIO')),
    acento: colorValido_(c['COLOR_ACENTO'], valorDefecto_('COLOR_ACENTO')),
    exito: colorValido_(c['COLOR_EXITO'], valorDefecto_('COLOR_EXITO')),
    fondo: colorValido_(c['COLOR_FONDO'], valorDefecto_('COLOR_FONDO')),
    texto: colorValido_(c['COLOR_TEXTO'], valorDefecto_('COLOR_TEXTO')),
    fuenteTitulos: fuenteTitulos,
    fuenteTexto: fuenteTexto,
    urlFuentes: 'https://fonts.googleapis.com/css2?' + familias.map(function (f) {
      return 'family=' + f.replace(/ /g, '+') + ':wght@400;500;600;700';
    }).join('&') + '&display=swap',
    logo: logo_('LOGO_ARCHIVO_ID'),
    logoColor: logo_('LOGO_COLOR_ARCHIVO_ID')
  };
}

/**
 * Logo como data URI (lo lee la cuenta dueña, así no depende de los permisos del visitante).
 * Si no se puede leer o es muy pesado, se usa LOGO_URL; sin eso, la página muestra el logotipo tipográfico.
 */
function logo_(claveConfig) {
  var c = config_();
  var id = aTexto_(c[claveConfig]);
  if (idDriveValido_(id)) {
    var cache = CacheService.getScriptCache();
    var clave = 'logo_' + id;
    var guardado = cache.get(clave);
    if (guardado) return guardado;
    try {
      var blob = DriveApp.getFileById(id).getBlob();
      var tipo = blob.getContentType();
      var bytes = blob.getBytes();
      if (/^image\//.test(tipo) && bytes.length < 600000) {
        var uri = 'data:' + tipo + ';base64,' + Utilities.base64Encode(bytes);
        if (uri.length < 95000) cache.put(clave, uri, 21600);
        return uri;
      }
    } catch (e) {
      // Se continúa con LOGO_URL.
    }
  }
  return urlSegura_(c['LOGO_URL']);
}
