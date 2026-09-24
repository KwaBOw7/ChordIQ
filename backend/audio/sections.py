"""
Song structure: parts (intro / verse / chorus / bridge ...), the lines
inside them, and how many times each one repeats and where.

Method (works on the BAR grid, so it follows the music, not the clock):
  1. One feature vector per bar: chroma + bass chroma + timbre (MFCC).
  2. Find the block length (4, 8 or 16 bars) and offset at which the song
     "tiles" best, i.e. blocks that line up with earlier/later blocks.
  3. Group blocks with the same material (average-linkage clustering).
     Consecutive equal blocks merge into one part. Labels A, B, C are
     reliable; names (Verse/Chorus/...) are heuristics from repetition
     count and loudness.
  4. Half-block "lines" are clustered song-wide too, so you get
     "line a appears 6 times, at ...".

Not solved here: repeats of individual LYRIC lines. That needs
speech/lyrics transcription, a separate feature.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from audio.features import HOP
from audio.key import key_at

BLOCK_LENGTHS = (8, 4, 16)
ENERGY_WEIGHT = 0.2   # how much loudness/arrangement level counts in CLUSTERING
                      # (kept separate from block-length detection, which is
                      # harmony-only: chord loops repeat even when a song
                      # builds, so tiling must not be thrown off by energy)
TIMBRE_WEIGHT = 0.2             # harmony leads; timbre (vocals, drums) only nudges
BLOCK_CLUSTER_DISTANCE = 0.16   # 1 - similarity below which blocks are "the same"
LINE_CLUSTER_DISTANCE = 0.16
UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOW = "abcdefghijklmnopqrstuvwxyz"


# ------------------------------------------------------------- bar features
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _bar_features(features, bars, key_data=None):
    sr, n = features["sr"], features["n_frames"]
    chroma, bass = features["chroma_smooth"], features["bass"]
    mfcc, rms = features["mfcc"], features["rms"]

    rows_c, rows_b, rows_t, energy = [], [], [], []
    for bar in bars:
        a = int(bar["start"] * sr / HOP)
        b = int(bar["end"] * sr / HOP)
        a = max(0, min(n - 1, a))
        b = max(a + 1, min(n, b))
        c_vec = np.median(chroma[:, a:b], axis=1)
        b_vec = np.mean(bass[:, a:b], axis=1)
        if key_data is not None:
            # Make harmony KEY-RELATIVE, so a chorus repeated a semitone
            # higher after a key change still matches the earlier choruses.
            tonic, _ = key_at(key_data, 0.5 * (bar["start"] + bar["end"]))
            c_vec = np.roll(c_vec, -int(tonic))
            b_vec = np.roll(b_vec, -int(tonic))
        rows_c.append(_unit(c_vec))
        rows_b.append(_unit(b_vec))
        rows_t.append(np.mean(mfcc[1:, a:b], axis=1))
        energy.append(float(np.mean(rms[a:b])))

    T = np.array(rows_t)
    T = (T - T.mean(axis=0)) / (T.std(axis=0) + 1e-9)
    T = T / np.sqrt(T.shape[1])

    F = np.hstack([np.array(rows_c), 0.5 * np.array(rows_b), TIMBRE_WEIGHT * T])
    return F, np.array(energy)


def _stack(F, a, b):
    v = F[a:b].reshape(-1)
    return _unit(v)


def _energy_features(F, energy):
    """
    F with a loudness/arrangement column appended, for use in CLUSTERING
    only (never in _choose_tiling). A long vamp with a static chord loop
    but a real intro -> full-arrangement build has near-identical harmony
    throughout; without this, that build collapses into one part.
    """
    z = (energy - energy.mean()) / (energy.std() + 1e-9)
    z = np.clip(z, -2.5, 2.5)
    return np.hstack([F, ENERGY_WEIGHT * z[:, None]])


# ----------------------------------------------------------------- tiling
def _tiling_score(F, L, o):
    N = len(F)
    starts = list(range(o, N - L + 1, L))
    if len(starts) < 3:
        return -1.0, starts
    B = np.array([_stack(F, s, s + L) for s in starts])
    S = B @ B.T
    np.fill_diagonal(S, -1.0)
    return float(S.max(axis=1).mean()), starts


def _choose_tiling(F):
    """Return (L, offset). Prefers longer blocks when they tile about as well."""
    N = len(F)
    results = {}
    for L in BLOCK_LENGTHS:
        best = (-1.0, 0)
        for o in range(L):
            score, _ = _tiling_score(F, L, o)
            # small preference for tilings that start at the beginning
            score -= 0.002 * o
            if score > best[0]:
                best = (score, o)
        results[L] = best

    valid = {L: v for L, v in results.items() if v[0] > 0}
    if not valid:
        return None, 0, {}
    top = max(v[0] for v in valid.values())
    # 8 bars is by far the most common part length in pop / gospel / soul.
    # Only leave it when another length tiles CLEARLY better.
    if 8 in valid and valid[8][0] >= top - 0.05:
        L = 8
    else:
        L = max(valid, key=lambda k: valid[k][0])
    return L, results[L][1], {k: round(v[0], 3) for k, v in results.items()}


# -------------------------------------------------------------- clustering
def _cluster_vectors(V, threshold):
    """Average-linkage clustering of unit vectors; labels by first appearance."""
    n = len(V)
    if n == 1:
        return [0]
    S = V @ V.T
    D = np.clip(1.0 - S, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    raw = fcluster(Z, t=threshold, criterion="distance")
    order, out = {}, []
    for l in raw:
        if l not in order:
            order[l] = len(order)
        out.append(order[l])
    return out


def _prefix_similarity(F, a0, a1, b0, b1):
    """Similarity of two possibly different-length bar ranges over their overlap."""
    L = min(a1 - a0, b1 - b0)
    if L <= 0:
        return 0.0
    return float(np.dot(_stack(F, a0, a0 + L), _stack(F, b0, b0 + L)))


# ------------------------------------------------------------------ naming
def _name_parts(part_clusters, part_spans, energy, n_bars):
    k = max(part_clusters) + 1
    counts = [part_clusters.count(c) for c in range(k)]
    cenergy = []
    for c in range(k):
        vals = [np.mean(energy[a:b]) for (a, b), cid in zip(part_spans, part_clusters) if cid == c]
        cenergy.append(float(np.mean(vals)))

    names, used = {}, set()

    first, last = part_clusters[0], part_clusters[-1]
    a0, b0 = part_spans[0]
    if counts[first] == 1 and (b0 - a0) <= max(8, 0.2 * n_bars) and len(part_spans) > 2:
        names[first] = "Intro"
        used.add(first)
    aL, bL = part_spans[-1]
    if last not in used and counts[last] == 1 and (bL - aL) <= max(12, 0.2 * n_bars) and len(part_spans) > 2:
        names[last] = "Outro"
        used.add(last)

    repeated = [c for c in range(k) if c not in used and counts[c] >= 2]
    if repeated:
        by_energy = sorted(repeated, key=lambda c: cenergy[c], reverse=True)
        if len(repeated) >= 2 and cenergy[by_energy[0]] > 1.05 * cenergy[by_energy[-1]]:
            names[by_energy[0]] = "Chorus"
            for i, c in enumerate(by_energy[1:]):
                names[c] = "Verse" if i == 0 else f"Section {UP[c]}"
        else:
            for c in repeated:
                names[c] = f"Section {UP[c]}"
        used.update(repeated)

    # exactly one Bridge: the unique middle part closest to ~70% through the song
    uniques = [c for c in range(k) if c not in used]
    bridge = None
    best_d = 1e9
    for c in uniques:
        pos = np.mean([(a + b) / 2 for (a, b), cid in zip(part_spans, part_clusters) if cid == c]) / n_bars
        if 0.4 <= pos <= 0.9 and abs(pos - 0.7) < best_d:
            bridge, best_d = c, abs(pos - 0.7)
    for c in uniques:
        names[c] = "Bridge" if c == bridge else f"Section {UP[c]}"
    return names, cenergy


def _merge_inseparable(parts):
    """
    If part X is ALWAYS followed by part Y and Y is ALWAYS preceded by X,
    they are really one longer part (e.g. a 16-bar verse made of two
    different 8-bar halves). Merge them into one.
    """
    changed = True
    while changed:
        changed = False
        seq = [p["cluster"] for p in parts]
        for x in sorted(set(seq)):
            idx = [i for i, c in enumerate(seq) if c == x]
            nxt = {seq[i + 1] if i + 1 < len(seq) else None for i in idx}
            if len(idx) < 2 or len(nxt) != 1:
                continue
            y = next(iter(nxt))
            if y is None or y == x:
                continue
            yidx = [i for i, c in enumerate(seq) if c == y]
            prev = {seq[i - 1] if i > 0 else None for i in yidx}
            if prev != {x} or len(yidx) != len(idx):
                continue
            # merge every (x, y) pair; the merged part reuses cluster id x
            merged, i = [], 0
            while i < len(parts):
                if seq[i] == x and i + 1 < len(parts) and seq[i + 1] == y:
                    merged.append({
                        "cluster": x,
                        "a": parts[i]["a"],
                        "b": parts[i + 1]["b"],
                        "blocks": parts[i]["blocks"] + parts[i + 1]["blocks"],
                    })
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
            changed = True
            break
    return parts


ESCALATED_TIMBRE_WEIGHT = 0.4   # used ONLY inside _split_long_parts, never globally
LONG_PART_MIN_BARS = 48        # a part shorter than this is never re-split
LONG_PART_RATIO = 3.0          # ...or shorter than this many times the median part


def _split_long_parts(parts, F, raw_timbre, energy, L, dist):
    """
    A part far longer than the rest of the song may be hiding real
    sub-structure (a vocalist change, a texture shift) that stayed under
    the radar at the weight tuned for the WHOLE song - timbre differences
    are real but subtle relative to how strongly identical harmony
    dominates the block vector. Rather than raising sensitivity
    everywhere (which was tested and found to break genuine repeat
    detection elsewhere), only the suspiciously long part is re-examined,
    at a higher timbre weight, using its OWN internal blocks. Everything
    already correctly clustered is left untouched.
    """
    if len(parts) < 2:
        return parts
    lengths = [p["b"] - p["a"] for p in parts]
    median_len = sorted(lengths)[len(lengths) // 2]
    threshold = max(LONG_PART_MIN_BARS, LONG_PART_RATIO * median_len)

    chroma_bass = F[:, :24]
    Fc_hi = np.hstack([chroma_bass, ESCALATED_TIMBRE_WEIGHT * raw_timbre])

    # Every local re-cluster below must get cluster ids that can NEVER
    # collide with any id used elsewhere in `parts` (or by another long
    # part split in this same call) - otherwise two musically unrelated
    # parts that happen to both start a fresh local id at 0 look, to the
    # caller, like the same repeated section by sheer coincidence.
    next_id = max((p["cluster"] for p in parts), default=-1) + 1

    out = []
    for p in parts:
        length = p["b"] - p["a"]
        n_blocks = length // L
        if length < threshold or n_blocks < 4:
            out.append(p)
            continue

        sub_starts = list(range(p["a"], p["b"] - L + 1, L))
        V = np.array([_stack(Fc_hi, a, a + L) for a in sub_starts])
        local_clusters = _cluster_vectors(V, dist)

        if max(local_clusters) == 0:   # escalation found nothing new
            out.append(p)
            continue

        sub_clusters = [next_id + c for c in local_clusters]
        next_id += max(local_clusters) + 1

        sub_parts = []
        for s, c in zip(sub_starts, sub_clusters):
            if sub_parts and sub_parts[-1]["cluster"] == c:
                sub_parts[-1]["b"] = s + L
                sub_parts[-1]["blocks"] += 1
            else:
                sub_parts.append({"cluster": c, "a": s, "b": s + L, "blocks": 1})
        remainder = p["b"] - sub_parts[-1]["b"]
        if remainder:
            sub_parts[-1]["b"] += remainder
            sub_parts[-1]["blocks"] += 1

        out.extend(_merge_inseparable(sub_parts))
    return out


# --------------------------------------------------------------------- main
def _progression(chords, t0, t1):
    return [c for c in chords if t0 - 1e-6 <= c["start"] < t1 - 1e-6]


def analyze_sections(features, beat_data, chord_sequence, key_data=None):
    bars = beat_data["bars"]
    N = len(bars)
    empty = {"sections": [], "structure": [], "lines": [], "form": "", "block_bars": None}
    if N < 8:
        return {**empty, "note": "Too few bars to find structure."}

    F, energy = _bar_features(features, bars, key_data)
    L, offset, tiling_scores = _choose_tiling(F)
    if L is None:
        return {**empty, "note": "Song does not repeat enough to find structure."}

    # Block LENGTH comes from harmony alone (F). Which blocks count as the
    # SAME PART also considers loudness/arrangement (Fc), so a static chord
    # loop that quietly builds into a full arrangement is not one giant part.
    Fc = _energy_features(F, energy)

    # ---- blocks ------------------------------------------------------
    starts = list(range(offset, N - L + 1, L))
    main_spans = [(s, s + L) for s in starts]
    Vb = np.array([_stack(Fc, a, b) for a, b in main_spans])
    main_clusters = _cluster_vectors(Vb, BLOCK_CLUSTER_DISTANCE)

    spans = list(main_spans)
    clusters = list(main_clusters)
    next_id = max(main_clusters) + 1

    def attach(a, b):
        nonlocal next_id
        best_c, best_s = None, 0.0
        if (b - a) < 0.75 * L:      # a short fragment is not a repeat of a full part
            next_id += 1
            return next_id - 1
        for (ma, mb), c in zip(main_spans, main_clusters):
            s = _prefix_similarity(Fc, a, b, ma, mb)
            if s > best_s:
                best_c, best_s = c, s
        if best_s >= 0.80:
            return best_c
        next_id += 1
        return next_id - 1

    head = (0, offset) if offset > 0 else None
    tail_start = starts[-1] + L if starts else 0
    tail = (tail_start, N) if N - tail_start >= 1 else None

    if head:
        spans.insert(0, head)
        clusters.insert(0, attach(*head))
    if tail:
        spans.append(tail)
        clusters.append(attach(*tail))

    # re-index clusters by first appearance
    remap, ordered = {}, []
    for c in clusters:
        if c not in remap:
            remap[c] = len(remap)
        ordered.append(remap[c])
    clusters = ordered

    # ---- merge consecutive equal blocks into parts -------------------
    parts = []   # each: dict(cluster, a, b, blocks)
    for (a, b), c in zip(spans, clusters):
        if parts and parts[-1]["cluster"] == c:
            parts[-1]["b"] = b
            parts[-1]["blocks"] += 1
        else:
            parts.append({"cluster": c, "a": a, "b": b, "blocks": 1})

    parts = _merge_inseparable(parts)

    raw_timbre = F[:, 24:] / TIMBRE_WEIGHT
    parts = _split_long_parts(parts, F, raw_timbre, energy, L, BLOCK_CLUSTER_DISTANCE)

    # keep cluster ids contiguous after merging
    remap2 = {}
    for p in parts:
        if p["cluster"] not in remap2:
            remap2[p["cluster"]] = len(remap2)
    for p in parts:
        p["cluster"] = remap2[p["cluster"]]

    part_clusters = [p["cluster"] for p in parts]
    part_spans = [(p["a"], p["b"]) for p in parts]
    names, cenergy = _name_parts(part_clusters, part_spans, energy, N)

    # ---- lines (half blocks), song-wide ------------------------------
    Ll = max(2, L // 2)
    line_spans = []
    s = offset % Ll
    if s > 0:
        line_spans.append((0, s))
    for a in range(s, N - Ll + 1, Ll):
        line_spans.append((a, a + Ll))
    if line_spans and line_spans[-1][1] < N and N - line_spans[-1][1] >= 1:
        line_spans.append((line_spans[-1][1], N))
    full_lines = [(a, b) for a, b in line_spans if b - a == Ll]
    Vl = np.array([_stack(Fc, a, b) for a, b in full_lines])
    line_cluster_of = {}
    if len(Vl) >= 2:
        lc = _cluster_vectors(Vl, LINE_CLUSTER_DISTANCE)
        for span, c in zip(full_lines, lc):
            line_cluster_of[span] = c
    else:
        for span in full_lines:
            line_cluster_of[span] = 0
    n_line_clusters = (max(line_cluster_of.values()) + 1) if line_cluster_of else 0
    for span in line_spans:
        if span not in line_cluster_of:
            # partial line at head/tail: match by prefix, else new cluster
            best_c, best_s = None, 0.0
            for fs, c in (line_cluster_of.items() if (span[1] - span[0]) >= 0.75 * Ll else []):
                sim = _prefix_similarity(Fc, span[0], span[1], fs[0], fs[1])
                if sim > best_s:
                    best_c, best_s = c, sim
            if best_s >= 0.80:
                line_cluster_of[span] = best_c
            else:
                line_cluster_of[span] = n_line_clusters
                n_line_clusters += 1

    # ---- vamps: long parts that are one loop repeated ------------------
    vamp_clusters = []
    for c in set(part_clusters):
        insts = [p for p in parts if p["cluster"] == c]
        vamp_like = True
        for p in insts:
            ls = [line_cluster_of[sp] for sp in line_spans if sp[0] >= p["a"] and sp[1] <= p["b"]]
            if p["b"] - p["a"] < 16 or not ls:
                vamp_like = False
                break
            top = max(ls.count(x) for x in set(ls))
            if top / len(ls) < 0.70:
                vamp_like = False
                break
        if vamp_like:
            vamp_clusters.append(c)
    for i, c in enumerate(sorted(vamp_clusters)):
        names[c] = "Vamp" if len(vamp_clusters) == 1 else f"Vamp {UP[c % 26]}"

    # ---- assemble output --------------------------------------------
    counts = {c: part_clusters.count(c) for c in set(part_clusters)}
    seen = {}
    sections = []
    for p in parts:
        c = p["cluster"]
        seen[c] = seen.get(c, 0) + 1
        t0, t1 = bars[p["a"]]["start"], bars[p["b"] - 1]["end"]
        prog = _progression(chord_sequence, t0, t1)
        lines = [
            LOW[line_cluster_of[sp] % 26]
            for sp in line_spans if sp[0] >= p["a"] and sp[1] <= p["b"]
        ]
        sections.append({
            "label": UP[c % 26],
            "name": names[c],
            "instance": seen[c],
            "of": counts[c],
            "start": float(t0),
            "end": float(t1),
            "bar_start": int(p["a"]),
            "bar_end": int(p["b"]),
            "bars": int(p["b"] - p["a"]),
            "blocks": int(p["blocks"]),
            "lines": lines,
            "chords": [x["chord"] for x in prog],
            "nashville": [x.get("nashville") for x in prog],
        })

    structure = []
    for c in sorted(counts):
        inst = [s_ for s_ in sections if s_["label"] == UP[c % 26]]
        structure.append({
            "label": UP[c % 26],
            "name": names[c],
            "occurrences": counts[c],
            "instances": [
                {"instance": i + 1, "start": s_["start"], "end": s_["end"], "bars": s_["bars"]}
                for i, s_ in enumerate(inst)
            ],
            "progression": inst[0]["chords"],
            "nashville": inst[0]["nashville"],
            "loudness": round(cenergy[c], 4),
        })

    lines_out = []
    by_cluster = {}
    for sp in line_spans:
        by_cluster.setdefault(line_cluster_of[sp], []).append(sp)
    for c in sorted(by_cluster):
        spans_c = by_cluster[c]
        first = spans_c[0]
        prog = _progression(chord_sequence, bars[first[0]]["start"], bars[first[1] - 1]["end"])
        inst = []
        for i, sp in enumerate(spans_c):
            sec = next(
                (s_ for s_ in sections if s_["bar_start"] <= sp[0] < s_["bar_end"]), None
            )
            inst.append({
                "instance": i + 1,
                "start": float(bars[sp[0]]["start"]),
                "end": float(bars[sp[1] - 1]["end"]),
                "part": (f"{sec['label']}{sec['instance']}" if sec else None),
            })
        lines_out.append({
            "label": LOW[c % 26],
            "bars": int(first[1] - first[0]),
            "occurrences": len(spans_c),
            "progression": [x["chord"] for x in prog],
            "nashville": [x.get("nashville") for x in prog],
            "instances": inst,
        })

    form = " - ".join(
        s_["name"] if not s_["name"].startswith("Section ") else s_["label"]
        for s_ in sections
    )
    form_letters = " ".join(s_["label"] for s_ in sections)

    return {
        "sections": sections,
        "structure": structure,
        "lines": lines_out,
        "form": form,
        "form_letters": form_letters,
        "block_bars": int(L),
        "line_bars": int(Ll),
        "grid_offset_bars": int(offset),
        "tiling_scores": tiling_scores,
    }
