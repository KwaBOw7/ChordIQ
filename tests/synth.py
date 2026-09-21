"""
Synthetic songs with a KNOWN answer key, for regression-testing the analyzer.

Real recordings are harder than these (vocals, reverb, ad-libs, extended
chords). Passing here proves the logic works; it does not prove accuracy on
real songs. Always check real songs by ear against a chart you trust.
"""
import numpy as np

SR = 22050
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _env(n, attack=0.01, decay=3.0):
    t = np.arange(n) / SR
    return np.minimum(1, t / attack) * np.exp(-decay * t)


def tone(freq, dur, amp=0.15):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * freq * k * t) for k, a in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)))
    return amp * y * _env(n, 0.02, 0.6)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def chord_audio(root_pc, quality, dur, loud=1.0):
    third = {"maj": 4, "min": 3}[quality[:3]]
    seventh = 10 if quality.endswith("7") else None
    notes = [60 + root_pc, 60 + root_pc + third, 60 + root_pc + 7]
    if seventh:
        notes.append(60 + root_pc + seventh)
    y = sum(tone(midi_hz(m), dur, 0.10 * loud) for m in notes)
    y = y + tone(midi_hz(36 + root_pc), dur, 0.25 * loud)  # bass
    return y


def kick(n=4000):
    t = np.arange(n) / SR
    f = 120 * np.exp(-t * 30) + 45
    return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18)


def snare(n=3000):
    rng = np.random.default_rng(1)
    t = np.arange(n) / SR
    return 0.35 * rng.standard_normal(n) * np.exp(-t * 35)


def hat(n=1200):
    rng = np.random.default_rng(2)
    t = np.arange(n) / SR
    return 0.12 * rng.standard_normal(n) * np.exp(-t * 90)


def place(buf, sample, t):
    i = int(t * SR)
    if 0 <= i < len(buf):
        j = min(len(buf), i + len(sample))
        buf[i:j] += sample[: j - i]


def drums(buf, start, n_bars, beat, meter, loud=1.0):
    """meter: '4/4' | '3/4' | '6/8' (beat = quarter, or dotted-quarter for 6/8)."""
    K, S, H = kick(), snare(), hat()
    per_bar = {"4/4": 4, "3/4": 3, "6/8": 2, "12/8": 4}[meter]
    for b in range(n_bars):
        t0 = start + b * per_bar * beat
        for i in range(per_bar):
            t = t0 + i * beat
            if meter == "4/4":
                place(buf, K if i in (0, 2) else S, t)
                place(buf, H, t)
                place(buf, H, t + beat / 2)
            elif meter == "3/4":
                place(buf, K if i == 0 else S * 0.6, t)
                place(buf, H, t)
                place(buf, H, t + beat / 2)
            elif meter == "12/8":  # 4 dotted-quarter beats, 3 eighths each
                place(buf, K if i in (0, 2) else S, t)
                for k in range(3):
                    place(buf, H, t + k * beat / 3)
            else:  # 6/8: two dotted-quarter beats, each with 3 eighths
                place(buf, K if i == 0 else S, t)
                for k in range(3):
                    place(buf, H, t + k * beat / 3)
        if loud > 1.0:
            place(buf, S * 0.5, t0 + beat * per_bar - beat / 2)


# (root offset from tonic, quality)
I, ii, iii, IV, V, vi = (0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min")
V7 = (7, "maj7")


def render(tonic=8, bpm=100, meter="4/4", structure=None, mod_semitones=0):
    """
    structure: list of (name, bars_per_chord, [chords], loud)
    Returns (audio, truth) where truth lists sections with time spans.
    """
    per_bar = {"4/4": 4, "3/4": 3, "6/8": 2, "12/8": 4}[meter]
    beat = 60.0 / bpm
    bar_len = per_bar * beat

    if structure is None:
        verse = [I, vi, IV, V]
        chorus = [IV, V, I, vi]
        bridge = [vi, IV, ii, V7]
        structure = [
            ("Intro", 1, [I, V], 1.0, 4),
            ("Verse", 2, verse, 1.0, 8),
            ("Chorus", 2, chorus, 1.4, 8),
            ("Verse", 2, verse, 1.0, 8),
            ("Chorus", 2, chorus, 1.4, 8),
            ("Bridge", 2, bridge, 1.1, 8),
            ("Chorus", 2, chorus, 1.4, 8),
            ("Outro", 1, [I], 0.8, 2),
        ]

    total_bars = sum(s[4] for s in structure)
    buf = np.zeros(int((total_bars * bar_len + 2.0) * SR))
    truth, truth_chords = [], []
    bar = 0
    for name, bars_per_chord, chords, loud, n_bars in structure:
        t0 = bar * bar_len
        # modulate the final chorus/outro up
        is_last_chorus = (name in ("Chorus", "Outro") and mod_semitones and bar >= total_bars - 8 - 2)
        tn = (tonic + (mod_semitones if is_last_chorus else 0)) % 12
        for b in range(n_bars):
            ch = chords[(b // bars_per_chord) % len(chords)]
            root = (tn + ch[0]) % 12
            q = ch[1]
            dur = bar_len
            place(buf, chord_audio(root, q, dur, loud), (bar + b) * bar_len)
            label = NAMES[root] + ("m" if q == "min" else ("maj7" if q == "maj7" else ""))
            truth_chords.append(((bar + b) * bar_len, ((bar + b) + 1) * bar_len, label, root, q))
        drums(buf, t0, n_bars, beat, meter, loud)
        truth.append({"name": name, "start": t0, "end": (bar + n_bars) * bar_len, "bars": n_bars})
        bar += n_bars

    buf = buf / (np.max(np.abs(buf)) + 1e-9) * 0.9
    return buf.astype(np.float32), {"sections": truth, "chords": truth_chords, "bpm": bpm, "meter": meter, "tonic": tonic}
