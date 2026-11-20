// Service worker de Cuadra: la interfaz abre al instante y funciona como app en PC y celular.
// Los datos (/api) nunca se guardan en caché: siempre vienen del servidor.
const VERSION = "cuadra-v1";
const BASE = ["/", "/css/app.css", "/js/app.js", "/js/api.js", "/js/ui.js", "/icono.svg", "/manifest.webmanifest", "/icono-192.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(BASE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((claves) => Promise.all(claves.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.startsWith("/api/")) return;
  if (e.request.mode === "navigate") {
    e.respondWith(fetch(e.request).catch(() => caches.match("/")));
    return;
  }
  // Recursos de la interfaz: se sirven desde la caché y se actualizan por detrás.
  e.respondWith(caches.open(VERSION).then(async (cache) => {
    const guardado = await cache.match(e.request);
    const red = fetch(e.request).then((r) => { if (r.ok) cache.put(e.request, r.clone()); return r; }).catch(() => guardado);
    return guardado || red;
  }));
});
