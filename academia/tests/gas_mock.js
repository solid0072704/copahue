/**
 * Simulador mínimo de Google Apps Script para probar la academia con Node (sin Google).
 * Carga los .gs de ../src en un contexto aislado con DriveApp, SpreadsheetApp, Session,
 * HtmlService, CacheService, PropertiesService, LockService, Utilities y ScriptApp simulados.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const SRC = path.join(__dirname, '..', 'src');

/* ------------------------------------------------------------------ Sheets */

class Rango {
  constructor(hoja, fila, col, nFilas, nCols) {
    Object.assign(this, { hoja, fila, col, nFilas, nCols });
    const proxy = new Proxy(this, {
      get(obj, prop) {
        if (prop in obj) {
          const v = obj[prop];
          return typeof v === 'function' ? v.bind(obj) : v;
        }
        return () => proxy; // setFontWeight, setBackground, setDataValidation…: sin efecto
      }
    });
    this._proxy = proxy;
    return proxy;
  }
  getValues() {
    const out = [];
    for (let r = 0; r < this.nFilas; r++) {
      const fila = this.hoja.datos[this.fila - 1 + r] || [];
      const vals = [];
      for (let c = 0; c < this.nCols; c++) {
        const v = fila[this.col - 1 + c];
        vals.push(v === undefined || v === null ? '' : v);
      }
      out.push(vals);
    }
    return out;
  }
  setValues(valores) {
    if (valores.length !== this.nFilas || valores.some(f => f.length !== this.nCols)) {
      throw new Error(`setValues: dimensiones ${valores.length}x${valores[0] && valores[0].length} ≠ ${this.nFilas}x${this.nCols}`);
    }
    valores.forEach((fila, r) => fila.forEach((v, c) => this.hoja.poner(this.fila + r, this.col + c, v)));
    return this._proxy;
  }
  setValue(v) { this.hoja.poner(this.fila, this.col, v); return this._proxy; }
}

class Hoja {
  constructor(nombre) { this.nombre = nombre; this.datos = []; }
  getName() { return this.nombre; }
  poner(fila, col, v) {
    while (this.datos.length < fila) this.datos.push([]);
    const f = this.datos[fila - 1];
    while (f.length < col) f.push('');
    f[col - 1] = v;
  }
  getRange(fila, col, nFilas = 1, nCols = 1) {
    if (fila < 1 || col < 1 || nFilas < 1 || nCols < 1) throw new Error(`getRange inválido (${fila},${col},${nFilas},${nCols})`);
    return new Rango(this, fila, col, nFilas, nCols);
  }
  getLastRow() {
    for (let i = this.datos.length - 1; i >= 0; i--) {
      if ((this.datos[i] || []).some(v => v !== '' && v !== null && v !== undefined)) return i + 1;
    }
    return 0;
  }
  getLastColumn() {
    let max = 0;
    this.datos.forEach(f => {
      for (let i = f.length - 1; i >= 0; i--) if (f[i] !== '' && f[i] !== undefined) { max = Math.max(max, i + 1); break; }
    });
    return max;
  }
  getMaxRows() { return Math.max(1000, this.datos.length); }
  setFrozenRows() { return this; }
  /** Utilidad de pruebas: filas como objetos. */
  objetos() {
    const [cab, ...resto] = this.datos;
    return resto.filter(f => f.some(v => v !== '')).map(f => Object.fromEntries(cab.map((c, i) => [c, f[i] === undefined ? '' : f[i]])));
  }
}

class Libro {
  constructor(id, nombre) { this.id = id; this.nombre = nombre; this.hojas = [new Hoja('Hoja 1')]; }
  getId() { return this.id; }
  getUrl() { return `https://docs.google.com/spreadsheets/d/${this.id}/edit`; }
  getName() { return this.nombre; }
  getSheetByName(n) { return this.hojas.find(h => h.nombre === n) || null; }
  insertSheet(n) { const h = new Hoja(n); this.hojas.push(h); return h; }
  getSheets() { return this.hojas.slice(); }
  deleteSheet(h) { this.hojas = this.hojas.filter(x => x !== h); }
}

/* ------------------------------------------------------------------ Drive */

class Iterador {
  constructor(lista) { this.lista = lista; this.i = 0; }
  hasNext() { return this.i < this.lista.length; }
  next() { return this.lista[this.i++]; }
}

function crearDrive(arbol) {
  // arbol: { carpetas: {id: {nombre, hijos: [ids], archivos: [ids]}}, archivos: {id: {nombre, mime, destino?, mimeDestino?, acceso, dueno, blob?}} }
  const archivo = (id) => {
    const a = arbol.archivos[id];
    if (!a) throw new Error('No se encontró el archivo ' + id);
    return {
      getId: () => id,
      getName: () => a.nombre,
      getMimeType: () => a.mime,
      getTargetId: () => a.destino || null,
      getTargetMimeType: () => a.mimeDestino || null,
      isTrashed: () => !!a.papelera,
      getSharingAccess: () => a.acceso || 'DOMAIN_WITH_LINK',
      getOwner: () => ({ getEmail: () => a.dueno || 'dueno@tuproyectodevida.cl' }),
      getBlob: () => ({ getContentType: () => a.mime, getBytes: () => a.bytes || [1, 2, 3] })
    };
  };
  const carpeta = (id) => {
    const c = arbol.carpetas[id];
    if (!c) throw new Error('No se encontró la carpeta ' + id);
    return {
      getId: () => id,
      getName: () => c.nombre,
      getFiles: () => new Iterador((c.archivos || []).map(archivo)),
      getFolders: () => new Iterador((c.hijos || []).map(carpeta)),
      getSharingAccess: () => c.acceso || 'DOMAIN_WITH_LINK',
      getOwner: () => ({ getEmail: () => 'dueno@tuproyectodevida.cl' })
    };
  };
  return { getFileById: archivo, getFolderById: carpeta };
}

/* ------------------------------------------------------------------ Plantillas HtmlService */

function compilarPlantilla(fuente) {
  let codigo = 'var __s = [];\n';
  const re = /<\?(!?=)?([\s\S]*?)\?>/g;
  let ultimo = 0;
  let m;
  while ((m = re.exec(fuente))) {
    codigo += '__s.push(' + JSON.stringify(fuente.slice(ultimo, m.index)) + ');\n';
    if (m[1] === '=') codigo += '__s.push(__escapar(' + m[2].trim().replace(/;$/, '') + '));\n';
    else if (m[1] === '!=') codigo += '__s.push(String(' + m[2].trim().replace(/;$/, '') + '));\n';
    else codigo += m[2] + '\n';
    ultimo = re.lastIndex;
  }
  codigo += '__s.push(' + JSON.stringify(fuente.slice(ultimo)) + ');\nreturn __s.join("");';
  return codigo;
}

function salidaHtml(contenido) {
  const salida = {
    contenido,
    titulo: '',
    metas: {},
    getContent: () => salida.contenido,
    setTitle: (t) => { salida.titulo = t; return salida; },
    addMetaTag: (nombre, valor) => { salida.metas[nombre] = valor; return salida; },
    /** Como la sirve Google: con <title> y las etiquetas meta agregadas por el servidor. */
    paginaCompleta: () => salida.contenido.replace('<head>', '<head>' +
      Object.entries(salida.metas).map(([n, v]) => `<meta name="${n}" content="${v}">`).join('') +
      (salida.titulo ? `<title>${salida.titulo.replace(/</g, '&lt;')}</title>` : '')),
    setXFrameOptionsMode: () => salida,
    setFaviconUrl: () => salida
  };
  return salida;
}

/* ------------------------------------------------------------------ Contexto */

function crearEntorno(opciones = {}) {
  const estado = {
    usuarioActivo: opciones.usuario || 'nueva.asesora@tuproyectodevida.cl',
    dueno: opciones.dueno || 'admin.academia@tuproyectodevida.cl',
    propiedades: {},
    cache: new Map(),
    libros: {},
    contadorLibros: 0,
    logs: [],
    triggers: [],
    arbol: opciones.arbol || { carpetas: {}, archivos: {} }
  };

  const ctx = {
    console,
    Session: {
      getActiveUser: () => ({ getEmail: () => estado.usuarioActivo }),
      getEffectiveUser: () => ({ getEmail: () => estado.dueno })
    },
    PropertiesService: {
      getScriptProperties: () => ({
        getProperty: (k) => (k in estado.propiedades ? estado.propiedades[k] : null),
        setProperty: (k, v) => { estado.propiedades[k] = String(v); }
      })
    },
    CacheService: {
      getScriptCache: () => ({
        get: (k) => (estado.cache.has(k) ? estado.cache.get(k) : null),
        put: (k, v) => { if (String(v).length > 100000) throw new Error('Valor de caché muy grande'); estado.cache.set(k, v); },
        remove: (k) => estado.cache.delete(k)
      })
    },
    LockService: { getScriptLock: () => ({ waitLock: () => {}, releaseLock: () => {} }) },
    SpreadsheetApp: {
      create: (nombre) => {
        const id = 'libro' + (++estado.contadorLibros) + 'xxxxxxxxxxxx';
        estado.libros[id] = new Libro(id, nombre);
        return estado.libros[id];
      },
      openById: (id) => {
        if (!estado.libros[id]) throw new Error('No existe la planilla ' + id);
        return estado.libros[id];
      },
      newDataValidation: () => {
        const b = new Proxy({}, { get: (o, p) => (p === 'build' ? () => ({}) : () => b) });
        return b;
      }
    },
    DriveApp: null,
    Utilities: {
      formatDate: (d) => d.toISOString().slice(0, 10).split('-').reverse().join('-'),
      base64Encode: (bytes) => Buffer.from(bytes).toString('base64')
    },
    ScriptApp: {
      getService: () => ({ getUrl: () => 'https://script.google.com/a/macros/tuproyectodevida.cl/s/PRUEBA/exec' }),
      getProjectTriggers: () => estado.triggers,
      deleteTrigger: (t) => { estado.triggers = estado.triggers.filter(x => x !== t); },
      newTrigger: (fn) => {
        const t = { getHandlerFunction: () => fn };
        const cadena = { timeBased: () => cadena, everyDays: () => cadena, atHour: () => cadena, create: () => { estado.triggers.push(t); return t; } };
        return cadena;
      }
    },
    Logger: { log: (m) => estado.logs.push(String(m)) }
  };
  ctx.DriveApp = crearDrive(estado.arbol);

  const leerArchivo = (nombre) => fs.readFileSync(path.join(SRC, nombre.endsWith('.html') ? nombre : nombre + '.html'), 'utf8');
  ctx.HtmlService = {
    createHtmlOutputFromFile: (nombre) => salidaHtml(leerArchivo(nombre)),
    createHtmlOutput: (html) => salidaHtml(html || ''),
    createTemplateFromFile: (nombre) => {
      const plantilla = {};
      Object.defineProperty(plantilla, 'evaluate', {
        enumerable: false,
        value: () => {
          const codigo = compilarPlantilla(leerArchivo(nombre));
          const vars = Object.assign({}, plantilla);
          const fn = vm.runInContext('(function(__v, __escapar){ with(__v){ ' + codigo + ' } })', contexto);
          return salidaHtml(fn(vars, (t) => contexto.escaparHtml_(t)));
        }
      });
      return plantilla;
    }
  };

  const contexto = vm.createContext(ctx);
  const archivos = fs.readdirSync(SRC).filter(f => f.endsWith('.gs')).sort();
  archivos.forEach(f => vm.runInContext(fs.readFileSync(path.join(SRC, f), 'utf8'), contexto, { filename: f }));

  /** Ejecuta una función del proyecto como lo haría una nueva ejecución de Apps Script (cachés en memoria limpios). */
  const ejecutar = (nombre, ...args) => {
    contexto.reiniciarCaches_();
    const resultado = contexto[nombre](...args);
    // google.script.run serializa las respuestas: se imita con JSON (y evita comparar objetos de otro «realm»).
    if (resultado === undefined || resultado === null || typeof resultado.getContent === 'function') return resultado;
    return JSON.parse(JSON.stringify(resultado));
  };
  const comoUsuario = (email) => { estado.usuarioActivo = email; };
  const libro = () => estado.libros[estado.propiedades.ACADEMIA_DB_ID];
  const hoja = (nombre) => libro().getSheetByName(nombre);

  return { contexto, estado, ejecutar, comoUsuario, libro, hoja };
}

module.exports = { crearEntorno, compilarPlantilla };
