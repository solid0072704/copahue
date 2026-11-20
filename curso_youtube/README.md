# Curso de YouTube · Análisis de estados financieros con Claude

Todo lo necesario para producir el curso de una hora: guion, láminas, caso práctico, narración y video.
El contenido es una sola fuente (`contenido/`) de la que salen el guion, las imágenes, el texto para la voz,
los subtítulos y los capítulos.

| Qué | Dónde |
|---|---|
| Guion para leer y corregir | `GUION.md` (se genera) |
| Narración y láminas, por módulo | `contenido/m00_intro.py` … `m12_cierre.py` |
| Caso práctico (Distribuidora Alerce, ficticia) | `caso.py` · todas las cifras se calculan y verifican aquí |
| Material descargable para la audiencia | `material/caso_distribuidora_alerce.xlsx` y `material/prompts.md` |
| Título, descripción, capítulos, etiquetas y miniatura | `youtube/publicacion.md` y `youtube/miniatura.png` |
| Registro de la narración generada (qué audio corresponde a cada segmento) | `narracion.json` |

## Cómo se produce el video

```bash
python -m curso_youtube.deck                       # duración estimada por módulo
python -m curso_youtube.guion                      # GUION.md y material/prompts.md
python -m curso_youtube.construir <trabajo>        # láminas en HTML y PNG de 1920 x 1080 (avisa si algo se sale)
python -m curso_youtube.guion --voz <trabajo>/voz  # un .txt por audio: el texto exacto que recibe la voz
#   opcional: --tono "[calm] [professional]" antepone una indicación de tono (ElevenLabs v4 no la lee en voz alta)
#   … generar cada audio con la voz clonada y guardarlo como <audio>/seg_XXX.mp3 …
python -m curso_youtube.construir <trabajo> --animar --audio <audio> --hilos 4 --reanudar   # animación cuadro a cuadro
python -m curso_youtube.ensamblar <trabajo> <audio> --salida curso.mp4 --fin 20
python -m curso_youtube.youtube curso_capitulos.txt  # descripción con los capítulos exactos
```

- **Diseño.** La identidad sale de las presentaciones del autor: azul noche, crema y dorado; Montserrat para
  títulos, Source Sans 3 para texto, Newsreader cursiva para el énfasis y Geist Mono para etiquetas (todas en
  `fuentes/`). Las portadas y aperturas de capítulo usan las imágenes de `imagenes/`; las láminas de contenido
  llevan al fondo una cinta de líneas finas que cambia levemente en cada capítulo.
- **Movimiento.** `movimiento.js` define cómo entra cada elemento: los títulos suben palabra por palabra, los
  filetes se dibujan, las cifras cuentan desde cero, los prompts se escriben, las barras y líneas de los gráficos
  crecen y el foco se desplaza por tablas y prompts. Es determinista (el mismo paso y el mismo instante dan
  siempre la misma imagen), así que `construir --animar` lo captura cuadro a cuadro a 30 fps; con `--audio`
  cada animación cabe en lo que dura su paso. `ensamblar.py` muestra la animación de cada paso y después su
  imagen final, funde una lámina con la siguiente y abre cada capítulo con un barrido. Sin `--animar`, el
  video se arma igual con imágenes fijas.
- **Si se interrumpe.** La captura (`--reanudar`) y la codificación (por tramos, uno por módulo) retoman donde
  quedaron: solo se repite lo que cambió.
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
  segmento (`seg_XXX`).
- **Audios.** No se guardan en el repositorio (pesan unos 54 MB). `narracion.json` anota la generación de
  ElevenLabs que corresponde a cada segmento, para volver a descargarlos desde la cuenta sin gastar créditos.

## Requisitos

Python 3.11+, `numpy`, `Pillow`, `xlsxwriter`, `ffmpeg` y Node con Playwright (Chromium) para capturar las láminas.
Las fuentes van incluidas en `fuentes/` (licencia OFL). La captura animada del curso completo ocupa unos 8 GB
de imágenes temporales en la carpeta de trabajo.

## Pruebas

```bash
pytest tests/test_curso_voz.py tests/test_curso_deck.py tests/test_curso_subtitulos.py tests/test_curso_video.py -q
```

Verifican que el caso cuadre (balance, flujo, patrimonio), que las cifras se lean bien en voz alta, que ningún
elemento quede sin aparecer, que la duración estimada esté cerca de una hora, que los subtítulos no partan
cifras ni se traslapen y que el video se arme completo con su animación y sus transiciones.
