# Curso de YouTube · Análisis de estados financieros con Claude

Todo lo necesario para producir el curso de una hora: guion, láminas, caso práctico, narración y video.
El contenido es una sola fuente (`contenido/`) de la que salen el guion, las imágenes, el texto para la voz,
los subtítulos y los capítulos.

| Qué | Dónde |
|---|---|
| Guion para leer y corregir | `GUION.md` (se genera) |
| Narración y láminas, por módulo | `contenido/m00_intro.py` … `m13_cierre.py` |
| Caso práctico (Distribuidora Alerce, ficticia) | `caso.py` · todas las cifras se calculan y verifican aquí |
| Material descargable para la audiencia | `material/caso_distribuidora_alerce.xlsx` y `material/prompts.md` |
| Título, descripción, capítulos, etiquetas y miniatura | `youtube/publicacion.md` y `youtube/miniatura.png` |
| Retrato del autor para la miniatura (viene de su flyer en Canva, exportado con fondo transparente) | `imagenes/autor.png` |
| Registro de la narración generada (qué audio corresponde a cada segmento) | `narracion.json` |

## Cómo se produce el video

```bash
python -m curso_youtube.deck                       # duración estimada por módulo
python -m curso_youtube.guion                      # GUION.md y material/prompts.md
python -m curso_youtube.construir <trabajo>        # láminas en HTML y PNG de 1920 x 1080 (avisa si algo se sale)
python -m curso_youtube.guion --voz <trabajo>/voz  # un .txt por audio: el texto exacto que recibe la voz
#   opcional: --tono "[calm] [professional]" antepone una indicación de tono (ElevenLabs v4 no la lee en voz alta)
#   … generar cada audio con la voz clonada y guardarlo como <audio>/seg_XXX.mp3 …
python -m curso_youtube.construir <trabajo> --continuo --audio <audio> --hilos 4 --reanudar  # el video de cada lámina
python -m curso_youtube.ensamblar <trabajo> <audio> --salida curso.mp4 --fin 20
python -m curso_youtube.youtube curso_capitulos.txt  # descripción con los capítulos exactos
```

- **Diseño.** La identidad sale de las presentaciones del autor: azul noche, crema y dorado; Montserrat para
  títulos, Source Sans 3 para texto, Newsreader cursiva para el énfasis y Geist Mono para etiquetas (todas en
  `fuentes/`). Las portadas y aperturas de capítulo usan las imágenes de `imagenes/`; las láminas de contenido
  llevan al fondo una cinta de líneas finas que cambia levemente en cada capítulo.
- **Movimiento.** El video no tiene pantallas quietas. `movimiento.js` reproduce cada lámina de corrido con tres
  capas: el fondo (la cinta de líneas ondula y unas motas suben, con el reloj del video, así que no saltan al
  cambiar de lámina); la entrada de cada elemento repartida por su paso (el título primero y el detalle cuando la
  narración lo desarrolla, las filas de una tabla en cascada, iconos que se dibujan, cifras que cuentan, barras y
  líneas que crecen, prompts que se escriben y un puntero que los envía, respuestas de Claude que se van
  generando); y lo que mantiene viva la pantalla entre un paso y otro (una luz que recorre lo que se está
  explicando, el cursor, el avance del capítulo en el pie). De una lámina a la siguiente lo anterior se va, y cada
  capítulo abre con un telón que cruza la pantalla. Todo es determinista: el mismo instante da siempre la misma
  imagen.
- **Captura.** `construir --continuo --audio <audio>` calcula con el audio cuántos cuadros dura cada lámina y
  cada paso, reproduce cada lámina cuadro a cuadro (30 por segundo) en Chromium y pasa los cuadros directo a
  ffmpeg: queda un video por escena en `<trabajo>/escenas`, sin guardar cuadros en disco, más una imagen por paso
  tomada del video en `escenas/vistas` para revisarlo. `ensamblar` une las escenas y comprueba que calcen con el
  audio. Sin `--continuo`, `construir` deja una imagen por paso y el video se arma con imágenes fijas.
- **Si se interrumpe.** `--reanudar` no repite las escenas ya hechas con el mismo HTML, estilo, motor y audio:
  la captura de una hora (unos 70 minutos con 4 navegadores) se retoma donde quedó.
- **Narración.** Cada lámina tiene `pasos`: un texto por cada cambio en pantalla. El guion se escribe con cifras
  («11.800 millones», «18 %»); `voz.py` las convierte a palabras y reescribe términos como «Claude» → «Clod»
  para que la voz en español los pronuncie bien.
- **Sincronía.** `ensamblar.py` reparte cada audio entre sus pasos según el largo del texto y ajusta cada cambio
  a una pausa real de la voz. También nivela el volumen (−16 LUFS) y genera subtítulos (`.srt`) y capítulos.
- **Subtítulos.** Una frase por subtítulo, en dos líneas de hasta 42 caracteres, con las cifras como se ven en
  pantalla. Cada cambio de frase se ajusta a la pausa real de la voz. `ensamblar … --anexos` los vuelve a
  escribir (junto con los capítulos) sin renderizar el video otra vez.
- **Borrador sin voz.** `ensamblar … --borrador` rellena con silencio los audios que falten, para revisar ritmo
  y duración antes de gastar créditos de voz.
- **Cambiar el texto.** Edita el archivo del módulo, vuelve a generar la lámina y regenera solo el audio de ese
  segmento (`seg_XXX`). Los segmentos se numeran en orden: si agregas o quitas láminas, los audios que siguen
  cambian de número (su texto no cambia, así que se renombran y no hay que generarlos de nuevo).
- **Índice y primeros pasos.** La introducción trae un índice (`tipo="indice"`): cada parte nombra sus
  capítulos, y `deck` le calcula los minutos y la barra de capítulos a escala. El capítulo `m01_primeros`
  explica dónde se usa Claude, cómo se instala y qué planes existen; sus datos (planes, precios de referencia y
  aplicaciones) se verificaron en claude.com/pricing y claude.com/download en la fecha que indica el archivo:
  conviene revisarlos antes de volver a publicar.
- **Audios.** No se guardan en el repositorio (pesan unos 54 MB). `narracion.json` anota la generación de
  ElevenLabs que corresponde a cada segmento, para volver a descargarlos desde la cuenta sin gastar créditos.

## Requisitos

Python 3.11+, `numpy`, `Pillow`, `xlsxwriter`, `ffmpeg` y Node con Playwright (Chromium) para capturar las láminas.
Las fuentes van incluidas en `fuentes/` (licencia OFL).

## Pruebas

```bash
pytest tests/test_curso_voz.py tests/test_curso_deck.py tests/test_curso_subtitulos.py tests/test_curso_video.py -q
```

Verifican que el caso cuadre (balance, flujo, patrimonio), que las cifras se lean bien en voz alta, que ningún
elemento quede sin aparecer, que la duración estimada esté cerca de una hora, que los subtítulos no partan
cifras ni se traslapen y que el video se arme completo con su animación y sus transiciones.
