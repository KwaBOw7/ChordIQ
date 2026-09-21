"""
Key detection.

* Global key from CQT chroma (tuning-corrected) with Temperley profiles.
* Key CHANGES over time (gospel/soul often modulate up a semitone or tone).
  Tracked with a Viterbi path over 12 "key regions" (a major key and its
  relative minor form one region), so relative major/minor never flickers.
  Mode (major/minor) is then decided once for the whole song.
* Correct flat/sharp spelling (Ab major, not G# major).
"""

import numpy as np

from audio.features import HOP

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Major-key tonics conventionally written with flats: F, Bb, Eb, Ab, Db
_FLAT_MAJOR_TONICS = {5, 10, 3, 8, 1}

# Temperley (2005) profiles - work better than Krumhansl on pop/rock/soul.
MAJOR_PROFILE = np.array(
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
)
MINOR_PROFILE = np.array(
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
)


# ------------------------------------------------------------------ spelling
def uses_flats(tonic_pc, mode):
    major_tonic = tonic_pc if mode == "major" else (tonic_pc + 3) % 12
    return major_tonic in _FLAT_MAJOR_TONICS


def note_names(tonic_pc, mode):
    """The 12 pitch-class names spelled for this key."""
    return FLAT_NAMES if uses_flats(tonic_pc, mode) else SHARP_NAMES


def name_of(pc, tonic_pc, mode):
    return note_names(tonic_pc, mode)[int(pc) % 12]


# ------------------------------------------------------------------ scoring
def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def _key_scores(chroma_vec):
    """Correlation with every major/minor profile. Returns (maj[12], min[12])."""
    maj = np.array([_corr(chroma_vec, np.roll(MAJOR_PROFILE, t)) for t in range(12)])
    mnr = np.array([_corr(chroma_vec, np.roll(MINOR_PROFILE, t)) for t in range(12)])
    return maj, mnr


def _region_scores(chroma_vec):
    """
    Score the 12 key regions. Region r = major key with tonic r
    (= relative minor with tonic r+9). Mode-agnostic.
    """
    maj, mnr = _key_scores(chroma_vec)
    return np.array([max(maj[r], mnr[(r + 9) % 12]) for r in range(12)])


def _normalise_frames(chroma):
    s = chroma.sum(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return chroma / s


# ----------------------------------------------------------------- timeline
def _viterbi_regions(scores, switch_cost, mod_cost):
    """
    scores: (W, 12). Maximise sum(scores) minus switching costs.
    Modulating up 1 or 2 semitones is cheaper (typical gospel key change).
    """
    W = scores.shape[0]
    cost = np.full((12, 12), switch_cost)
    for a in range(12):
        cost[a, a] = 0.0
        cost[a, (a + 1) % 12] = mod_cost
        cost[a, (a + 2) % 12] = mod_cost

    delta = np.zeros((W, 12))
    back = np.zeros((W, 12), dtype=int)
    delta[0] = scores[0]

    for t in range(1, W):
        cand = delta[t - 1][:, None] - cost  # (from, to)
        back[t] = np.argmax(cand, axis=0)
        delta[t] = cand[back[t], np.arange(12)] + scores[t]

    path = np.zeros(W, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(W - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def analyze_key(features, window_s=12.0, hop_s=3.0):
    sr = features["sr"]
    chroma = _normalise_frames(features["chroma_smooth"])
    n = chroma.shape[1]
    frame_s = HOP / sr
    duration = n * frame_s

    # Global candidates (for display)
    total = chroma.mean(axis=1)
    g_maj, g_min = _key_scores(total)
    candidates = []
    for t in range(12):
        candidates.append({"key": SHARP_NAMES[t], "mode": "major", "score": float(g_maj[t])})
        candidates.append({"key": SHARP_NAMES[t], "mode": "minor", "score": float(g_min[t])})
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Windowed region scores
    win = max(1, int(window_s / frame_s))
    hop = max(1, int(hop_s / frame_s))
    starts = list(range(0, max(1, n - win + 1), hop))

    win_chroma = []
    scores = np.zeros((len(starts), 12))
    for i, s in enumerate(starts):
        v = chroma[:, s:s + win].mean(axis=1)
        win_chroma.append(v)
        scores[i] = _region_scores(v)

    path = _viterbi_regions(scores, switch_cost=1.2, mod_cost=0.8)

    # Dominant region (most windows)
    region = int(np.argmax(np.bincount(path, minlength=12)))

    # Decide mode ONCE, from windows in the dominant region
    idx = [i for i, r in enumerate(path) if r == region]
    v = np.mean([win_chroma[i] for i in idx], axis=0)
    maj, mnr = _key_scores(v)
    major_score = float(maj[region])
    minor_score = float(mnr[(region + 9) % 12])
    mode = "major" if major_score >= minor_score else "minor"
    tonic = region if mode == "major" else (region + 9) % 12

    def region_to_tonic(r):
        return r if mode == "major" else (r + 9) % 12

    # Segments: each window "owns" the middle hop_s seconds around its centre
    segments = []
    W = len(path)
    for i, r in enumerate(path):
        centre = starts[i] * frame_s + window_s / 2.0
        t0 = 0.0 if i == 0 else centre - hop_s / 2.0
        t1 = duration if i == W - 1 else centre + hop_s / 2.0
        tn = region_to_tonic(int(r))
        if segments and segments[-1]["tonic"] == tn:
            segments[-1]["end"] = float(t1)
        else:
            segments.append({
                "start": float(t0),
                "end": float(t1),
                "tonic": int(tn),
                "mode": mode,
                "key": name_of(tn, tn, mode),
            })

    modulations = []
    for a, b in zip(segments, segments[1:]):
        modulations.append({
            "time": float(b["start"]),
            "from": f"{a['key']} {mode}",
            "to": f"{b['key']} {mode}",
            "semitones": int((b["tonic"] - a["tonic"]) % 12),
        })

    rel_tonic = (tonic + 3) % 12 if mode == "minor" else (tonic + 9) % 12
    rel_mode = "major" if mode == "minor" else "minor"

    return {
        "key": name_of(tonic, tonic, mode),
        "tonic": int(tonic),
        "mode": mode,
        "confidence": float(max(major_score, minor_score)),
        "margin": float(abs(major_score - minor_score)),
        "relative": {"key": name_of(rel_tonic, rel_tonic, rel_mode), "mode": rel_mode},
        "candidates": candidates,
        "timeline": segments,
        "modulations": modulations,
    }


def key_at(key_data, time_s):
    """(tonic_pc, mode) active at time_s."""
    tl = key_data.get("timeline") or []
    for seg in tl:
        if seg["start"] <= time_s < seg["end"]:
            return seg["tonic"], seg["mode"]
    if tl:
        return tl[-1]["tonic"], tl[-1]["mode"]
    return key_data["tonic"], key_data["mode"]


# --------------------------------------------------------- compat wrappers
def analyze_global_key(y, sample_rate, features=None):
    if features is None:
        from audio.features import extract_features
        features = extract_features(y, sample_rate)
    return analyze_key(features)


def detect_key(y, sample_rate):
    return analyze_global_key(y, sample_rate)
