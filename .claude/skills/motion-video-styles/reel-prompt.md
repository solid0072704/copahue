# Reel prompt (ES / EN)

A fill-in-the-blanks prompt to make a short vertical video with an AI coding agent and the
[`motion-video-styles`](../skill/motion-video-styles/SKILL.md) skill. Replace every `[PLACEHOLDER]`.
Nothing here depends on a specific voice, account or machine: pick your own TTS, brand and music
(see `skill/motion-video-styles/references/`).

Install the skill first (see the [README](../README.md#install-the-skill)), then paste one of the prompts below.

## Español

```
Usá la skill motion-video-styles. Quiero un reel vertical (1080×1920, 45–60 s) sobre [TEMA]
para [PÚBLICO], estilo [ESTILO]. Muestra [PRODUCTO]; el link para el final es [URL_PÚBLICA].

Estructura en cinco pasos:
1. Gancho: a quién le habla y qué necesita, en la primera frase.
2. Promesa breve: qué va a ver y qué se lleva.
3. Dos o tres preguntas concretas que el video responde después.
4. Demo: causa y efecto en la misma pantalla; un solo momento fuerte.
5. Recompensa que se pueda usar (una receta, un prompt, un recurso) y recién después el llamado a la acción.

Antes de escribir escenas, pasame el guion y una hoja de beats (qué pasa en cada palabra de la voz).
Después: renderizá 2–4 cuadros de prueba y mostrámelos, revisá que ningún texto se salga o se pise
y que no haya tramos quietos, corregí lo que haga falta y recién ahí renderizá el MP4.

Entregables: MP4 con subtítulos, portada (el cuadro del gancho sin subtítulos) y el texto del post.
No inventes datos ni capturas: si algo es ilustrativo, rotulalo.
```

## English

```
Use the motion-video-styles skill. I want a vertical reel (1080×1920, 45–60 s) about [TOPIC]
for [AUDIENCE], in the [STYLE] style. It shows [PRODUCT]; the link for the end is [PUBLIC_URL].

Five-step structure:
1. Hook: who it is for and what they need, in the first sentence.
2. Short promise: what they will see and take away.
3. Two or three concrete questions the video answers later.
4. Demo: cause and effect on the same screen; one single peak moment.
5. A usable reward (a recipe, a prompt, a resource), and only then the call to action.

Before writing scenes, give me the script and a beat sheet (what happens on each word of the voice).
Then: render 2–4 test stills and show them to me, check that no text overflows or overlaps and that
nothing stays still for long, fix what fails, and only then render the MP4.

Deliverables: MP4 with captions, a cover (the hook frame without captions) and the post text.
Don't invent data or screenshots: label anything illustrative.
```

## Filled example (fictional)

| placeholder | example |
|---|---|
| `[TEMA]` / `[TOPIC]` | exportar notas a PDF en 10 segundos |
| `[PÚBLICO]` / `[AUDIENCE]` | estudiantes que toman apuntes en el celular |
| `[ESTILO]` / `[STYLE]` | `notebook` (one of: pastel, stamp, notebook, desktop, neon-night, poster, blueprint) |
| `[PRODUCTO]` / `[PRODUCT]` | NotaFácil, a made-up notes app |
| `[URL_PÚBLICA]` / `[PUBLIC_URL]` | https://example.com |

Tip: the skill asks for your brand once (name, colors, logo, voice) and saves it in `brand.json`,
so later videos only need the five placeholders.
