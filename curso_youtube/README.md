# Curso de YouTube · Análisis de estados financieros con Claude

Todo lo necesario para producir el curso de una hora: guion, láminas, caso práctico, narración y video.
El contenido es una sola fuente (`contenido/`) de la que salen el guion, las imágenes, el texto para la voz,
los subtítulos y los capítulos.

| Qué | Dónde |
|---|---|
| Guion para leer y corregir | `GUION.md` (se genera) |
| Narración y láminas, por módulo | `contenido/m00_intro.py` … `m11_cierre.py` |
| Caso práctico (Distribuidora Alerce, ficticia) | `caso.py` · todas las cifras se calculan y verifican aquí |
| Material descargable para la audiencia | `material/caso_distribuidora_alerce.xlsx` y `material/prompts.md` |
| Título, descripción, capítulos, etiquetas y miniatura | `youtube/publicacion.md` y `youtube/miniatura.png` |

## Cómo se produce el video

```bash
python -m curso_youtube.deck                       # duración estimada por módulo
python -m curso_youtube.guion                      # GUION.md y material/prompts.md
python -m curso_youtube.construir <trabajo>        # láminas en HTML y PNG de 1920 x 1080 (avisa si algo se sale)
python -m curso_youtube.guion --voz <trabajo>/voz  # un .txt por audio: el texto exacto que recibe la voz
#   … generar cada audio con la voz clonada y guardarlo como <audio>/seg_XXX.mp3 …
python -m curso_youtube.ensamblar <trabajo> <audio> --salida curso.mp4 --fin 20
python -m curso_youtube.youtube curso_capitulos.txt  # descripción con los capítulos exactos
```

- **Narración.** Cada lámina tiene `pasos`: un texto por cada cambio en pantalla. El guion se escribe con cifras
  («11.800 millones», «18 %»); `voz.py` las convierte a palabras y reescribe términos como «Claude» → «Clod»
  para que la voz en español los pronuncie bien.
- **Sincronía.** `ensamblar.py` reparte cada audio entre sus pasos según el largo del texto y ajusta cada cambio
  a una pausa real de la voz. También nivela el volumen (−16 LUFS) y genera subtítulos (`.srt`) y capítulos.
- **Borrador sin voz.** `ensamblar … --borrador` rellena con silencio los audios que falten, para revisar ritmo
  y duración antes de gastar créditos de voz.
- **Cambiar el texto.** Edita el archivo del módulo, vuelve a generar la lámina y regenera solo el audio de ese
  segmento (`seg_XXX`).

## Requisitos

Python 3.11+, `numpy`, `Pillow`, `xlsxwriter`, `ffmpeg` y Node con Playwright (Chromium) para capturar las láminas.
Fuentes: Inter, Inter Display y JetBrains Mono.

## Pruebas

```bash
pytest tests/test_curso_voz.py tests/test_curso_deck.py -q
```

Verifican que el caso cuadre (balance, flujo, patrimonio), que las cifras se lean bien en voz alta, que ningún
elemento quede sin aparecer y que la duración estimada esté cerca de una hora.
