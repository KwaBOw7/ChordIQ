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


_NAME_TO_PC = {}
for _i, (_s, _f) in enumerate(zip(SHARP_NAMES, FLAT_NAMES)):
    _NAME_TO_PC[_s] = _i
    _NAME_TO_PC[_f] = _i
del _i, _s, _f


def parse_key(key_str):
    """'Ab major' / 'F# minor' / 'Bb' (defaults to major) -> (tonic_pc, mode)."""
    if not key_str:
        return None
    parts = key_str.strip().split()
    if not parts:
        return None
    note = parts[0][0].upper() + parts[0][1:] if len(parts[0]) > 1 else parts[0].upper()
    if note not in _NAME_TO_PC:
        return None
    mode = "minor" if len(parts) > 1 and parts[1].lower().startswith("min") else "major"
    return _NAME_TO_PC[note], mode


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


def analyze_key(features, window_s=12.0, hop_s=3.0, forced_key=None):
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
        candidates.append({"key": SHARP_NAMES[t], "tonic": t, "mode": "major", "score": float(g_maj[t])})
        candidates.append({"key": SHARP_NAMES[t], "tonic": t, "mode": "minor", "score": float(g_min[t])})
    candidates.sort(key=lambda c: c["score"], reverse=True)

    forced = parse_key(forced_key) if isinstance(forced_key, str) else forced_key
    if forced is not None:
        f_tonic, f_mode = forced
        rel_tonic = (f_tonic + 3) % 12 if f_mode == "minor" else (f_tonic + 9) % 12
        rel_mode = "major" if f_mode == "minor" else "minor"
        runner_up = next(
            (c for c in candidates if c["tonic"] != f_tonic), None
        )
        return {
            "key": name_of(f_tonic, f_tonic, f_mode),
            "tonic": int(f_tonic),
            "mode": f_mode,
            "confidence": float(next(
                (c["score"] for c in candidates if c["tonic"] == f_tonic and c["mode"] == f_mode),
                0.0,
            )),
            "margin": None,
            "user_forced": True,
            "relative": {"key": name_of(rel_tonic, rel_tonic, rel_mode), "mode": rel_mode},
            "runner_up": None,
            "candidates": [{k: v for k, v in c.items() if k != "tonic"} for c in candidates],
            "timeline": [{
                "start": 0.0, "end": float(duration),
                "tonic": int(f_tonic), "mode": f_mode,
                "key": name_of(f_tonic, f_tonic, f_mode),
            }],
            "modulations": [],
        }

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

    # Best candidate on a DIFFERENT tonic (the real ambiguity: tonic vs
    # fifth, relative major/minor, etc.) is a much more useful "runner-up"
    # than the second row of the raw list, which is often the same tonic's
    # other mode.
    top_score = next(c["score"] for c in candidates if c["tonic"] == tonic and c["mode"] == mode)
    runner_up = next((c for c in candidates if c["tonic"] != tonic), None)
    runner_up_out = None
    if runner_up is not None and top_score > 0:
        gap = (top_score - runner_up["score"]) / top_score
        if gap < 0.15:   # within 15% of the winner: worth flagging as ambiguous
            runner_up_out = {
                "key": name_of(runner_up["tonic"], runner_up["tonic"], runner_up["mode"]),
                "mode": runner_up["mode"],
                "score": runner_up["score"],
            }

    return {
        "key": name_of(tonic, tonic, mode),
        "tonic": int(tonic),
        "mode": mode,
        "confidence": float(max(major_score, minor_score)),
        "margin": float(abs(major_score - minor_score)),
        "user_forced": False,
        "relative": {"key": name_of(rel_tonic, rel_tonic, rel_mode), "mode": rel_mode},
        "runner_up": runner_up_out,
        "candidates": [{k: v for k, v in c.items() if k != "tonic"} for c in candidates],
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

# --------------------------------------------------- chord-based refinement
#
# The note-profile Viterbi above decides MAJOR vs its RELATIVE MINOR using
# only which of the 12 pitch classes are present, in what proportion. A
# song in Ab major and one in F minor can look almost identical that way,
# since they share the same 7 notes. Chords resolve this: a I chord that
# is actually played major-vs-minor, how much time is spent ON the tonic
# chord, and V->I motion are much stronger evidence for "home" than note
# content alone. This is exactly the method a musician uses by ear.
#
# Scope: this ONLY re-examines relative major vs relative minor (the one
# ambiguity that is a safe, invertible relabeling of the same timeline).
# It does NOT re-examine unrelated candidates (e.g. the dominant, or a
# neighbouring key on the circle of fifths) - that would require redoing
# the whole key timeline, not just relabelling it.

_MAJOR_FAMILY = {0: {"major"}, 2: {"minor"}, 4: {"minor"}, 5: {"major"},
                 7: {"major"}, 9: {"minor"}, 10: {"major"}}
_MINOR_FAMILY = {0: {"minor"}, 3: {"major"}, 5: {"minor"},
                 7: {"major", "minor"}, 8: {"major"}, 10: {"major"}}


def _chord_family(quality):
    return "major" if quality in ("major", "major7", "dominant7") else "minor"


def _key_chord_score(chords, tonic, mode):
    table = _MAJOR_FAMILY if mode == "major" else _MINOR_FAMILY
    total = sum(c["duration"] for c in chords) or 1.0
    fit = tonic_time = 0.0
    resolutions = 0
    for c in chords:
        deg = (c["root_pc"] - tonic) % 12
        fam = _chord_family(c["quality"])
        if deg in table and fam in table[deg]:
            fit += c["duration"]
        if deg == 0 and fam == mode:
            tonic_time += c["duration"]
    for a, b in zip(chords, chords[1:]):
        if (a["root_pc"] - tonic) % 12 == 7 and (b["root_pc"] - tonic) % 12 == 0:
            resolutions += 1
    n = max(1, len(chords))
    combined = (
        0.55 * (fit / total)
        + 0.35 * (tonic_time / total)
        + 0.10 * min(1.0, (resolutions / n) * 4)
    )
    return {
        "diatonic_fit": fit / total,
        "tonic_time": tonic_time / total,
        "resolutions": resolutions,
        "combined": combined,
    }


def _endpoint_evidence(features, tonic):
    """
    How present a pitch class is at the very start and end of the
    recording, weighted toward the end (cadences are the strongest signal
    of the tonal center). Relative major/minor keys share almost the same
    overall note content, so a global histogram alone can't reliably tell
    them apart; a long vamp on the vi chord can even make it look like the
    WRONG one wins on raw duration. Endings settle the question the way a
    listener does - where the music actually resolves - independent of
    how much time was spent elsewhere.
    """
    chroma = features["chroma_smooth"]
    sr, hop, n = features["sr"], features["hop"], features["n_frames"]
    frame_s = hop / sr
    duration = n * frame_s
    if duration <= 2.0:
        return 0.0

    def window_profile(t0, t1):
        a = max(0, int(t0 / frame_s))
        b = min(n, int(t1 / frame_s))
        if b <= a:
            return None
        v = chroma[:, a:b].sum(axis=1)
        s = v.sum()
        return v / s if s > 0 else None

    early = window_profile(0.0, min(45.0, duration * 0.12))
    late = window_profile(max(0.0, duration - min(20.0, duration * 0.08)), duration)

    e = float(early[tonic]) if early is not None else 0.0
    l = float(late[tonic]) if late is not None else 0.0
    return 0.35 * e + 0.65 * l


def refine_key_mode(key_data, chords, features=None, margin=0.05):
    """
    Re-decide major vs relative-minor using chord evidence instead of note
    content. Returns a NEW key_data dict (timeline/modulations relabelled
    consistently) if chord evidence decisively disagrees; otherwise returns
    key_data unchanged (plus a diagnostic 'chord_evidence' field).
    """
    if not chords or key_data.get("user_forced"):
        return key_data

    tonic, mode = key_data["tonic"], key_data["mode"]
    region_tonic = tonic if mode == "major" else (tonic + 3) % 12
    rel_tonic = (region_tonic + 9) % 12
    major_key = (region_tonic, "major")
    minor_key = (rel_tonic, "minor")

    scores = {major_key: _key_chord_score(chords, *major_key),
              minor_key: _key_chord_score(chords, *minor_key)}

    # Endpoint evidence, when available, is folded in as its own term. It
    # exists specifically to counteract the duration bias above (a vamp
    # can make the wrong chord look "most tonic" by raw time-on-chord).
    if features is not None:
        for key in (major_key, minor_key):
            scores[key]["endpoint"] = _endpoint_evidence(features, key[0])
            scores[key]["combined"] += 0.35 * scores[key]["endpoint"]

    current = scores[(tonic, mode)]
    other_key = minor_key if (tonic, mode) == major_key else major_key
    other = scores[other_key]

    out = dict(key_data)
    out["chord_evidence"] = {"kept": current, "alternative": other}
    out["mode_corrected_by_chords"] = False

    if other["combined"] <= current["combined"] + margin:
        return out

    new_tonic, new_mode = other_key

    def region_of(t, m):
        return t if m == "major" else (t + 3) % 12

    def apply_mode(r, m):
        return r if m == "major" else (r + 9) % 12

    new_segments = []
    for seg in key_data["timeline"]:
        r = region_of(seg["tonic"], mode)
        nt = apply_mode(r, new_mode)
        ns = dict(seg)
        ns["tonic"], ns["mode"] = nt, new_mode
        ns["key"] = name_of(nt, nt, new_mode)
        new_segments.append(ns)

    new_modulations = []
    for a, b in zip(new_segments, new_segments[1:]):
        new_modulations.append({
            "time": b["start"],
            "from": f"{a['key']} {new_mode}",
            "to": f"{b['key']} {new_mode}",
            "semitones": int((b["tonic"] - a["tonic"]) % 12),
        })

    rel_t = (new_tonic + 3) % 12 if new_mode == "minor" else (new_tonic + 9) % 12
    rel_m = "major" if new_mode == "minor" else "minor"

    out.update({
        "key": name_of(new_tonic, new_tonic, new_mode),
        "tonic": int(new_tonic),
        "mode": new_mode,
        "relative": {"key": name_of(rel_t, rel_t, rel_m), "mode": rel_m},
        "timeline": new_segments,
        "modulations": new_modulations,
        "mode_corrected_by_chords": True,
        "chord_evidence": {"kept": other, "alternative": current},
    })
    return out
