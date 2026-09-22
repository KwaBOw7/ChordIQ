import sys
import time
from pathlib import Path

import numpy as np

# Allow both `python backend/main.py` and `uvicorn api:app` from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio.loader import load_audio
from audio.features import extract_features
from audio.key import analyze_key, refine_key_mode
from audio.beats import analyze_beats
from audio.chords import analyze_chords
from audio.progression import analyze_progression
from audio.sections import analyze_sections


BASE_DIR = Path(__file__).resolve().parent.parent
SONGS_DIR = BASE_DIR / "songs"

SUPPORTED = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


def find_song():
    """Find the first supported audio file in songs/."""
    if not SONGS_DIR.exists():
        raise FileNotFoundError(f"Songs folder not found: {SONGS_DIR}")

    files = sorted(
        p for p in SONGS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not files:
        raise FileNotFoundError(f"No supported audio files found in {SONGS_DIR}")
    return files[0]


def _clean(obj):
    """Make everything JSON-serialisable (numpy -> python)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def analyze_song(song_path, verbose=True, meter=None, bpm=None, key=None):
    def log(msg):
        if verbose:
            print(msg, flush=True)

    t0 = time.time()

    log("Loading audio...")
    y, sr = load_audio(song_path)
    duration = len(y) / sr

    log("Extracting features (this is the slow step)...")
    features = extract_features(y, sr)

    log("Analyzing key and key changes...")
    key_data = analyze_key(features, forced_key=key)

    log("Analyzing tempo, time signature and bars...")
    beat_data = analyze_beats(y, sr, features, forced_meter=meter, forced_bpm=bpm)

    log("Analyzing chords...")
    chord_data = analyze_chords(features, beat_data, key_data)

    log("Checking major vs relative-minor against the chords...")
    refined_key_data = refine_key_mode(key_data, chord_data)
    if refined_key_data["mode_corrected_by_chords"]:
        log(f"  -> corrected: {key_data['key']} {key_data['mode']} -> "
            f"{refined_key_data['key']} {refined_key_data['mode']}")
        key_data = refined_key_data
        chord_data = analyze_chords(features, beat_data, key_data)
    else:
        key_data = refined_key_data

    log("Analyzing chord progression...")
    progression = analyze_progression(chord_data, key_data["key"], key_data["mode"])

    log("Analyzing song structure...")
    structure = analyze_sections(features, beat_data, progression["sequence"], key_data)

    log(f"Done in {time.time() - t0:.1f}s")

    result = {
        "duration": duration,
        "sample_rate": sr,
        "tuning_cents": features["tuning"] * 100.0,

        "tempo": beat_data["tempo"],
        "measured_bpm": beat_data["measured_bpm"],
        "tempo_stability": beat_data["tempo_stability"],
        "tempo_user_forced": beat_data["tempo_user_forced"],
        "bar_rescue": beat_data.get("bar_rescue"),
        "beats": beat_data["beat_times"],

        "time_signature": beat_data["time_signature"],
        "bars": beat_data["bars"],

        "key": key_data["key"],
        "mode": key_data["mode"],
        "key_confidence": key_data["confidence"],
        "key_margin": key_data["margin"],
        "relative_key": key_data["relative"],
        "key_candidates": key_data["candidates"],
        "key_user_forced": key_data["user_forced"],
        "key_runner_up": key_data.get("runner_up"),
        "key_mode_corrected": key_data.get("mode_corrected_by_chords", False),
        "key_chord_evidence": key_data.get("chord_evidence"),
        "key_timeline": key_data["timeline"],
        "modulations": key_data["modulations"],

        "chords": progression["sequence"],
        "patterns": progression["patterns"],

        "sections": structure.get("sections", []),
        "structure": structure.get("structure", []),
        "form": structure.get("form", ""),
        "form_letters": structure.get("form_letters", ""),
        "lines": structure.get("lines", []),
        "block_bars": structure.get("block_bars"),
        "line_bars": structure.get("line_bars"),
    }
    return _clean(result)


def _fmt_time(t):
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}"


def main():
    song_path = find_song()
    print(f"Song: {song_path.name}")
    r = analyze_song(song_path)

    ts = r["time_signature"]
    print("\nRESULT\n======")
    print(f"Duration:        {r['duration']:.1f}s")
    print(f"Tempo:           {r['tempo']:.1f} BPM (stability {r['tempo_stability']:.3f}; lower = steadier)")
    print(f"Time signature:  {ts['label']}  (confidence {ts['confidence']:.2f}; "
          f"alternatives: {', '.join(a['label'] for a in ts['alternatives'][1:])})")
    print(f"Key:             {r['key']} {r['mode']}  (confidence {r['key_confidence']:.2f}; "
          f"relative {r['relative_key']['key']} {r['relative_key']['mode']})")
    for m in r["modulations"]:
        print(f"  Key change at {_fmt_time(m['time'])}: {m['from']} -> {m['to']} (+{m['semitones']} semitones)")

    print(f"\nFORM: {r['form']}")
    print(f"      {r['form_letters']}")
    print("\nSECTIONS")
    print("========")
    for s in r["sections"]:
        lines = " ".join(s["lines"])
        print(f"{_fmt_time(s['start'])} - {_fmt_time(s['end'])}  {s['label']}  {s['name']} "
              f"({s['instance']}/{s['of']}, {s['bars']} bars)  lines: {lines}")
        print(f"    {' | '.join(s['chords'][:16])}{' ...' if len(s['chords']) > 16 else ''}")

    print("\nREPEATED BLOCKS")
    print("===============")
    for st in r["structure"]:
        times = ", ".join(_fmt_time(i["start"]) for i in st["instances"])
        print(f"{st['label']} ({st['name']}): x{st['occurrences']} at {times}")
        print(f"    {' - '.join(st['progression'])}")
        print(f"    {' - '.join(str(n) for n in st['nashville'])}")

    print("\nLINES (half-blocks, repeated song-wide)")
    print("=======================================")
    for ln in r["lines"]:
        if ln["occurrences"] < 2:
            continue
        where = ", ".join(f"{i['part']}@{_fmt_time(i['start'])}" for i in ln["instances"])
        print(f"line {ln['label']} ({ln['bars']} bars) x{ln['occurrences']}: {' - '.join(ln['progression'])}")
        print(f"    {where}")

    print(f"\nChords detected: {len(r['chords'])}  |  Bars: {len(r['bars'])}")


if __name__ == "__main__":
    main()
