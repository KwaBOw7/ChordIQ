"""
Chord recognition (major, minor, dominant 7, major 7, minor 7).

Why the old detector produced a new chord every beat and lots of "dim":
it picked the best template independently on every beat, with no memory
and a diminished template that happily matches any noisy chroma.

This version:
  * uses tuning-corrected CQT chroma (beat-synchronous, median-pooled),
  * scores 5 chord types x 12 roots against templates + bass evidence,
  * adds a local-key prior (diatonic chords are more likely),
  * decodes with a Viterbi/HMM where a chord change is CHEAP on a downbeat
    and EXPENSIVE off the beat, so chords line up with bars.

Limits (be honest with users): extended chords (9ths, 11ths, 13ths, slash
chords, altered chords) are collapsed to the nearest of the 5 types.
"""

import numpy as np

from audio.key import key_at, name_of

QUALITIES = ["major", "minor", "dominant7", "major7", "minor7"]

# semitone offsets from root and their weights
_TEMPLATE_SPEC = {
    "major":     {0: 1.0, 4: 0.85, 7: 0.75},
    "minor":     {0: 1.0, 3: 0.85, 7: 0.75},
    "dominant7": {0: 1.0, 4: 0.85, 7: 0.75, 10: 0.60},
    "major7":    {0: 1.0, 4: 0.85, 7: 0.75, 11: 0.60},
    "minor7":    {0: 1.0, 3: 0.85, 7: 0.75, 10: 0.60},
}

# Small penalty so a 7th chord is only chosen when the 7th is really there.
_QUALITY_BIAS = {
    "major": 0.0, "minor": 0.0,
    "dominant7": -0.035, "major7": -0.045, "minor7": -0.035,
}

EMISSION_TEMPERATURE = 0.04   # cosine difference worth 1 nat
BASS_WEIGHT = 0.08

# Probability of a chord change at a given beat (per beat, before spreading
# over the other chords). Downbeat changes are common, off-beat ones rare.
P_CHANGE_DOWNBEAT = 0.45
P_CHANGE_MIDBAR = 0.12
P_CHANGE_OTHER = 0.02


def _build_templates():
    states = []
    mats = []
    for root in range(12):
        for q in QUALITIES:
            v = np.zeros(12)
            for off, w in _TEMPLATE_SPEC[q].items():
                v[(root + off) % 12] = w
            v /= np.linalg.norm(v)
            states.append((root, q))
            mats.append(v)
    return states, np.array(mats)  # (S, 12)


STATES, TEMPLATES = _build_templates()
STATE_BIAS = np.array([_QUALITY_BIAS[q] for _, q in STATES])


# ----------------------------------------------------------------- key prior
_MAJOR_DIATONIC = {
    0: ("major", "major7", "dominant7"),   # I  (I7 = bluesy, weaker)
    2: ("minor", "minor7"),                # ii
    4: ("minor", "minor7"),                # iii
    5: ("major", "major7"),                # IV
    7: ("major", "dominant7"),             # V
    9: ("minor", "minor7"),                # vi
}
_MAJOR_BORROWED = {10: ("major", "dominant7")}    # bVII

_MINOR_DIATONIC = {
    0: ("minor", "minor7"),                # i
    3: ("major", "major7"),                # bIII
    5: ("minor", "minor7"),                # iv
    7: ("minor", "minor7", "major", "dominant7"),  # v / V
    8: ("major", "major7"),                # bVI
    10: ("major", "dominant7"),            # bVII
}


def _key_prior(tonic, mode):
    """Log-bonus per state for a given key."""
    diatonic = _MAJOR_DIATONIC if mode == "major" else _MINOR_DIATONIC
    borrowed = _MAJOR_BORROWED if mode == "major" else {}
    bonus = np.zeros(len(STATES))
    for i, (root, q) in enumerate(STATES):
        deg = (root - tonic) % 12
        if deg in diatonic and q in diatonic[deg]:
            bonus[i] = 0.7
            if q in ("dominant7",) and deg == 0:
                bonus[i] = 0.3
        elif deg in borrowed and q in borrowed[deg]:
            bonus[i] = 0.3
        elif deg in diatonic or deg in borrowed:
            bonus[i] = 0.25          # right root, unusual quality
    return bonus


_PRIOR_CACHE = {}


def _prior_for(tonic, mode):
    key = (tonic, mode)
    if key not in _PRIOR_CACHE:
        _PRIOR_CACHE[key] = _key_prior(tonic, mode)
    return _PRIOR_CACHE[key]


# ------------------------------------------------------------------- viterbi
def _viterbi(emission, p_change):
    T, S = emission.shape
    log_stay = np.log(1.0 - p_change)
    log_move = np.log(p_change / (S - 1))

    delta = emission[0].copy()
    back = np.zeros((T, S), dtype=np.int32)
    idx = np.arange(S)

    for t in range(1, T):
        best_state = int(np.argmax(delta))
        stay = delta + log_stay[t]
        move = delta[best_state] + log_move[t]
        choose_stay = stay >= move
        new = np.where(choose_stay, stay, move)
        back[t] = np.where(choose_stay, idx, best_state)
        delta = new + emission[t]

    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# ---------------------------------------------------------------------- main
def analyze_chords(features, beat_data, key_data):
    """
    features  : output of extract_features
    beat_data : output of analyze_beats
    key_data  : output of analyze_key
    Returns a list of chord dicts (time-ordered, merged).
    """
    beat_times = np.asarray(beat_data["beat_times"])
    bchroma = beat_data["beat_chroma"]          # (12, B)
    bbass = beat_data["beat_bass"]              # (12, B)
    bpb = beat_data["time_signature"]["beats_per_bar"]
    phase = beat_data["downbeat_phase"]

    B = bchroma.shape[1]
    if B < 4:
        return []

    # Beat chroma -> unit vectors
    X = bchroma / (np.linalg.norm(bchroma, axis=0, keepdims=True) + 1e-9)
    bass = bbass / (bbass.sum(axis=0, keepdims=True) + 1e-9)

    # Cosine similarity with every chord template, plus bass evidence
    cos = X.T @ TEMPLATES.T                       # (B, S)
    roots = np.array([r for r, _ in STATES])
    bass_root = bass[roots, :].T                  # (B, S)
    score = cos + BASS_WEIGHT * bass_root + STATE_BIAS[None, :]

    # Key prior, using the key that is active at each beat
    prior = np.zeros_like(score)
    for i in range(B):
        tonic, mode = key_at(key_data, float(beat_times[i]))
        prior[i] = _prior_for(tonic, mode)

    emission = (score - score.max(axis=1, keepdims=True)) / EMISSION_TEMPERATURE + prior

    # Beats with almost no harmonic energy (silence): don't force a decision
    energy = bchroma.sum(axis=0)
    quiet = energy < 0.05 * np.median(energy)
    emission[quiet] = 0.0

    # Change probability depends on position in the bar
    pos = (np.arange(B) - phase) % bpb
    p_change = np.full(B, P_CHANGE_OTHER)
    p_change[pos == 0] = P_CHANGE_DOWNBEAT
    if bpb % 2 == 0 and bpb >= 4:
        p_change[pos == bpb // 2] = P_CHANGE_MIDBAR

    path = _viterbi(emission, p_change)

    # Beat-level -> merged chord segments
    end_time = beat_times[-1] + (beat_times[-1] - beat_times[-2])
    bounds = np.append(beat_times, end_time)

    chords = []
    for i in range(B):
        root, q = STATES[path[i]]
        conf = float(cos[i, path[i]])
        if (
            chords
            and chords[-1]["root_pc"] == root
            and chords[-1]["quality"] == q
        ):
            c = chords[-1]
            c["end"] = float(bounds[i + 1])
            c["beat_end"] = i + 1
            c["_conf"].append(conf)
        else:
            chords.append({
                "root_pc": int(root),
                "quality": q,
                "start": float(bounds[i]),
                "end": float(bounds[i + 1]),
                "beat_start": i,
                "beat_end": i + 1,
                "_conf": [conf],
            })

    for c in chords:
        mid = 0.5 * (c["start"] + c["end"])
        tonic, mode = key_at(key_data, mid)
        c["key_tonic"] = int(tonic)
        c["key_mode"] = mode
        c["root"] = name_of(c["root_pc"], tonic, mode)
        c["duration"] = c["end"] - c["start"]
        c["confidence"] = float(np.mean(c.pop("_conf")))
        c["bar"] = int((c["beat_start"] - phase) // bpb) if c["beat_start"] >= phase else -1

    return chords
