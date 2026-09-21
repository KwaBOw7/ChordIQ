# ChordIQ

Upload a song and get its **key** (including key changes), **tempo**, **time signature**, **chord progression**, and **song structure**: which parts repeat, how many times, and where.

Built first for gospel, pop, R&B and soul. Hip-hop is planned for later.

## What it produces

- **Key:** major/minor, correct flat/sharp spelling, relative key, and key changes over time (e.g. a chorus lifted up a semitone)
- **Tempo and time signature:** 4/4, 3/4, 6/8, 12/8 and others, with confidence and alternatives
- **Chords:** major, minor, dominant 7, major 7, minor 7, with Nashville numbers relative to the current key
- **Structure:** parts labelled A, B, C with names (Intro, Verse, Chorus, Bridge, Vamp, Outro), repeat counts, and the repeated 4-bar "lines" inside them
- **Overrides:** set the real tempo or time signature when the auto-detection is wrong

## How it works

1. Audio is split into harmonic and percussive parts, and chroma features are extracted once and shared.
2. Key is found by comparing note content to major/minor profiles, tracked over time with a Viterbi path.
3. Beats are tracked, and the bar length is cross-checked against chord-change and accent periodicity, which corrects trackers that lock onto a dotted-eighth pulse.
4. Chords are decoded with an HMM in which changes are cheap on downbeats and expensive elsewhere.
5. Structure comes from bar-level similarity: the song is tiled into blocks, matching blocks are clustered, and repeats are counted.

## Project layout

```
backend/    FastAPI app + analysis (backend/audio/)
frontend/   React + Vite interface
tests/      Synthetic songs with known answers (regression tests)
songs/      Put test audio here (ignored by git)
```

## Setup

Requires Python 3.10+, Node 18+, and (recommended) [ffmpeg](https://ffmpeg.org) for files with mislabeled formats.

```bash
# Backend
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cd backend
uvicorn api:app --reload --port 8000

# Frontend (second terminal)
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

Command-line mode (analyzes the first file in `songs/`):

```bash
python backend/main.py
```

## API

- `GET /health`
- `POST /analyze` with a multipart `file`. Optional query parameters: `meter` (e.g. `6/8`) and `bpm` (e.g. `70`) to override auto-detection.

## Tests

```bash
python tests/run_synth.py
```

Generates synthetic songs (4/4, 3/4, 12/8, and a key change) and compares the analysis to the known answer.

## Known limitations

- Extended chords (9ths, 13ths, slash chords) are simplified to the five supported types.
- 6/8 vs 12/8 is inherently ambiguous; very fast or very slow songs can be read at half or double tempo. Use the overrides.
- Section names (Verse/Chorus/Bridge) are heuristics; the A/B/C labels and repeat counts are more reliable.
- "Lines" are musical phrases, not lyric lines. Lyric-level repeats need lyrics transcription.
- Accuracy on real recordings has only been checked by ear on a small number of songs. Synthetic tests prove the logic, not real-world accuracy.
- A 5-minute song takes roughly 30 to 60 seconds to analyze.

## Roadmap

- Key override and runner-up key display
- Dockerfile and a job queue for deployment
- Lyrics transcription for line-level repeats
- Hip-hop support
