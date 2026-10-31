# Direction, revisions and QA — what the numbers don't catch

Lessons from producing and revising real reels with this kit. Read before writing a script and again
before every revision round.

## Script: hook, loops, reward

1. **Hook = who it is for + what they need, in the first sentence.** "If you make content and want a
   video like this one, stay and I'll show you how." Not a greeting, not a mystery.
2. **Short promise:** what they will see and take away.
3. **Two or three micro-loops, each one a concrete question with a concrete answer on screen:**
   "what did I give it?" → *this* prompt, readable, before anything runs; "how does it build the scenes?"
   → the construction happening; "how does it get to the final video?" → the preview, the check and the
   export command. Three titles shown at the start are not loops: open each one with the voice and close
   it visibly (tick, answer, the object that proves it).
4. **Demo with visible cause and effect.** If something is the result of a request, show the request
   first (sent, received, processed), then the execution, then the result. An object that appears and
   disappears without a reason breaks the chain: keep the same object and give it a reason to leave
   (it was sent, it docked, it became the next step).
5. **Reward before the call to action:** the prompt, recipe or resource the viewer can use. An install
   button alone is not a reward. The CTA comes after, in words, **without emoji** by default.

Say "final video", not "M P 4" over and over; the UI can still show "MP4".

## Never a long empty screen

Motion audits pass on camera drift alone. Check the frames yourself: if the hero object is empty while
the voice promises something (an empty phone, a grey panel), fill it with something real — a mirror of the
video's own draft, the tools arriving, the request being written. Measure it (how many sampled frames show
real content) instead of trusting "0 % still".

## Layout: air, not just safe zones

Text inside safe zones is the minimum. Also measure the gap between groups at the exact times they meet
(list ↔ preview, preview ↔ panel): 40 px or more. When a big element leaves, use the space it frees
instead of keeping everything glued at the top. No permanent grey disclaimer at the bottom edge: label
reconstructions inside the UI they describe and keep the details in a sources file.

## Sound

Voice in front (−16 LUFS), music low with ducking (−35 LUFS). Typing, keystroke and terminal tick sounds
get tiring fast: leave them **off** by default and keep only effects that mark a real event (an object
lands, a result appears, the peak moment).

## Voice: keep what was approved

- **Revisions start by recovering the approved takes.** Never regenerate the whole voice because the design
  changed. List which takes you keep and which sentences you really need to regenerate, then regenerate
  only those.
- **Join takes in pauses, never inside a syllable.** Find the lowest-energy point between two words in
  both takes and crossfade 20 ms there.
- **Check the first word.** Silence trimming can eat a soft first consonant ("S", "F"): the first 10 ms
  should be silence, not speech. Trim gently (e.g. −60 dB) and keep ~80 ms before the onset.
- **No aggressive tempo changes or pitch tricks** to make a line fit; re-time the animation instead.
- **Whisper and acoustic numbers are hints, not a verdict.** A transcript can say "Reel" while a person
  hears "real". If you cannot listen, say so, pick by transcript + pitch/rate close to the approved takes,
  and hand the alternatives to the person who decides. Natural, consistent voice beats any metric.
- TTS seeds may not be reproducible: save every take with its transcript the moment it is generated.

## Revision rounds

- **First a draft of one version** (one language, one ending). Expand to other languages and variants
  only after it is approved.
- Each round goes in a **new folder/slug**; approved files stay untouched (keep their checksums).
- Change the minimum that the feedback asks for; when a round loses strength, go back to the version that
  worked and carry over only the clear visual gains.
- Loop: fix specific points → stills strip (6 frames) → layout check → render → final strip and checks
  (decode, audio = video length, loudness, privacy). Don't spend hours on QA loops that cannot see what
  the person complained about.

## Privacy, on every surface

Video frames, voice, captions, terminal replays, preview frames inside the video, cover, post text,
prompts and example files: no personal names, emails, absolute paths, tokens, local endpoints, session
IDs or voice-clone names. Check text (the page's DOM over time) **and** pixels (OCR of rendered frames
and the cover), and report only counts.
