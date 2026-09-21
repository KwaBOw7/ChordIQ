"""Run the analyzer on synthetic songs and compare with the known truth."""
import sys, tempfile, os
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth import render, SR, NAMES
from main import analyze_song
from audio.harmony import NOTE_TO_SEMITONE


def chord_accuracy(result, truth):
    """Fraction of truth-time covered by the correct chord (root + major/minor)."""
    ok = total = 0.0
    for (t0, t1, label, root, q) in truth["chords"]:
        for c in result["chords"]:
            lo, hi = max(t0, c["start"]), min(t1, c["end"])
            if hi > lo:
                total += hi - lo
                want_minor = q == "min"
                got_minor = "minor" in c["quality"]
                pc = NOTE_TO_SEMITONE[c["root"]]
                if pc == root and want_minor == got_minor:
                    ok += hi - lo
    return ok / total if total else 0.0


def run(name, **kw):
    audio, truth = render(**kw)
    path = os.path.join(tempfile.gettempdir(), f"synth_{name}.wav")
    sf.write(path, audio, SR)
    r = analyze_song(path, verbose=False)
    print(f"\n=== {name} ===")
    print(f"truth : {NAMES[truth['tonic']]} , {truth['bpm']} bpm, {truth['meter']}")
    print(f"found : {r['key']} {r['mode']}, {r['tempo']:.1f} bpm, {r['time_signature']['label']} "
          f"(conf {r['time_signature']['confidence']:.2f}, alts {[a['label'] for a in r['time_signature']['alternatives'][1:]]})")
    print(f"key changes: {[(round(m['time'],1), m['from'], m['to']) for m in r['modulations']]}")
    print(f"chord accuracy (root+quality, time-weighted): {chord_accuracy(r, truth)*100:.0f}%")
    print("truth sections:", " | ".join(f"{s['name']}[{s['start']:.0f}-{s['end']:.0f}s]" for s in truth["sections"]))
    print("found sections:", " | ".join(f"{s['label']}:{s['name']}[{s['start']:.0f}-{s['end']:.0f}s]" for s in r["sections"]))
    print("form:", r["form"])
    return r


if __name__ == "__main__":
    which = sys.argv[1:] or ["44", "34", "68", "mod"]
    if "44" in which:
        run("4-4_Ab_100bpm", tonic=8, bpm=100, meter="4/4")
    if "34" in which:
        run("3-4_D_90bpm", tonic=2, bpm=90, meter="3/4")
    if "68" in which:
        run("6-8_G_60bpm", tonic=7, bpm=60, meter="6/8")
    if "128" in which:
        run("12-8_Eb_58bpm", tonic=3, bpm=58, meter="12/8")
    if "mod" in which:
        run("4-4_Ab_up1", tonic=8, bpm=96, meter="4/4", mod_semitones=1)
