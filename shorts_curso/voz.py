#!/usr/bin/env python3
"""Sincroniza la voz en off con las escenas.
Entrada: audio/crudo/sN.(wav|mp3) (narración completa de cada Short) y, abajo, los silencios
que separan las escenas. Salida: audio/short-N.m4a y tiempos.js (duración real de cada escena
y anclas para que cada elemento entre cuando se nombra)."""
import json, re, subprocess, wave
import numpy as np

SR = 48000
# Duraciones de diseño de cada escena (las mismas de contenido.js)
DISENO = {
    1: [4.0, 6.6, 3.0, 7.8, 5.0, 6.8, 4.6],
    2: [4.4, 3.7, 3.7, 3.7, 3.9, 7.4, 4.6],
    3: [4.6, 6.8, 7.2, 8.2, 4.2, 4.6],
    4: [4.6, 2.8, 3.3, 3.3, 2.7, 2.7, 2.9, 7.0, 4.6],
}
# Silencio (inicio, fin) entre una escena y la siguiente, en segundos de la narración
CORTES = {
    1: [(5.78, 6.28), (15.73, 16.21), (17.36, 17.85), (30.55, 31.05), (37.91, 38.49), (45.39, 45.96)],
    2: [(5.90, 6.39), (11.33, 11.80), (15.53, 15.99), (19.00, 19.50), (24.29, 24.87), (34.24, 34.74)],
    3: [(5.84, 6.44), (14.04, 14.62), (28.16, 28.76), (41.77, 42.31), (46.79, 47.27)],
    4: [(7.04, 7.49), (10.48, 11.02), (15.58, 16.05), (20.25, 20.74), (22.48, 23.02), (25.04, 25.54),
        (28.13, 28.58), (39.15, 39.61)],
}
# Anclas por escena: (segundo de diseño del elemento, segundo de la narración en que se nombra)
ANCLAS = {
    1: [[(1.0, 3.14), (1.9, 4.54)],
        [(1.9, 10.22), (3.3, 12.24)],
        [],
        [(0.4, 20.39), (2.3, 23.86), (4.3, 26.76)],
        [(1.9, 33.21)],
        [(0.7, 40.41), (4.1, 43.2), (4.6, 43.5)],
        []],
    2: [[(1.7, 3.94)],
        [(0.2, 7.22), (1.1, 9.61)],
        [(0.2, 12.73), (1.1, 14.23)],
        [(0.2, 16.84)],
        [(0.2, 20.38), (1.1, 22.25)],
        [(1.0, 27.54), (2.5, 30.20), (4.0, 31.53), (4.7, 32.9)],
        []],
    3: [[(1.9, 4.09)],
        [(0.7, 7.93), (4.3, 13.2)],
        [(0.4, 16.70), (1.7, 20.05), (3.0, 22.93), (4.3, 26.15)],
        [(0.5, 31.13), (2.2, 34.85), (3.9, 39.12)],
        [(0.3, 43.66), (1.1, 44.90)],
        []],
    4: [[(1.0, 2.3), (2.4, 4.50)],
        [],
        [(0.2, 11.87), (1.0, 12.76)],
        [(0.2, 17.00), (1.0, 18.38)],
        [(0.2, 21.64)],
        [(0.2, 23.94)],
        [(0.2, 26.52)],
        [(0.7, 29.61), (4.9, 38.2)],
        []],
}
COLA = 1.2          # segundos de cierre después de la última palabra
MINIMO = 0.75       # una escena nunca dura menos que esta fracción de su diseño


def leer(n):
    for ext in ('mp3',):
        try:
            raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f'audio/crudo/s{n}.{ext}', '-ac', '1',
                                  '-ar', str(SR), '-f', 's16le', '-'], capture_output=True, check=True).stdout
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
        except subprocess.CalledProcessError:
            continue
    raise SystemExit(f'falta audio/crudo/s{n}')


tiempos = {}
for n in sorted(DISENO):
    x = leer(n)
    fin = len(x) / SR
    c = [0.0] + [(a + b) / 2 for a, b in CORTES[n]] + [fin]
    partes, escenas = [], []
    for i, d0 in enumerate(DISENO[n]):
        seg = x[int(c[i] * SR):int(c[i + 1] * SR)]
        ultimo = i == len(DISENO[n]) - 1
        d = max(len(seg) / SR + (COLA if ultimo else 0), MINIMO * d0)
        d = round(d * 30) / 30                      # múltiplo de un cuadro
        pad = np.zeros(max(0, int(round(d * SR)) - len(seg)), dtype=np.float32)
        partes.append(np.concatenate([seg, pad])[:int(round(d * SR))])
        anclas = [[0.1, 0.1]] if ANCLAS[n][i] and ANCLAS[n][i][0][0] > 0.1 else []
        anclas += [[t, round(real - c[i], 3)] for t, real in ANCLAS[n][i]]
        escenas.append({'d': round(d, 4), 'a': anclas})
    y = np.concatenate(partes)
    pico = float(np.abs(y).max())
    y = y * min(1.0, 0.89 / pico) if pico > 0 else y
    w = wave.open(f'audio/short-{n}.wav', 'wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', f'audio/short-{n}.wav', '-af', 'loudnorm=I=-15:TP=-1.5:LRA=11',
                    '-ar', str(SR), '-ac', '2', '-c:a', 'aac', '-b:a', '192k', f'audio/short-{n}.m4a'], check=True)
    subprocess.run(['rm', f'audio/short-{n}.wav'])
    tiempos[n] = escenas
    print(n, round(len(y) / SR, 2), [e['d'] for e in escenas])

open('tiempos.js', 'w').write('// Generado por voz.py: duración real de cada escena y anclas [diseño, real]\nwindow.TIEMPOS = '
                              + re.sub(r'"(\w)":', r'\1:', json.dumps(tiempos)) + ';\n')
