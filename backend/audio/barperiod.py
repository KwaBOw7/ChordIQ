"""
Bar length from the music itself (harmonic rhythm + accents), independent of
the beat tracker.

Why: beat trackers lock onto whatever pulse is most regular. In worship and
gospel that is often a dotted-eighth (3-3-2 accents inside 4/4), giving a
"tempo" 4/3 too fast and bars of the wrong length. Chords, however, change
on the bar, and the kick/bass repeat on the bar, so a time-domain
autocorrelation of (chord-change + accent) finds the bar directly.

Used only as a CHECK on the tracker: it takes over when the two bar lengths
clearly disagree (not when they differ by a factor of 2, which is ordinary
1-bar vs 2-bar ambiguity).
"""

import numpy as np
from scipy.signal import find_peaks

from audio.features import HOP

MIN_BAR_S = 1.4
MAX_BAR_S = 9.0
QUARTER_MIN_S = 0.40     # 150 bpm
QUARTER_MAX_S = 1.20     # 50 bpm
MIN_STRENGTH = 0.18


def _z(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def harmonic_novelty(chroma, fs, win_s=0.5):
    """Chord-change indicator: cosine distance between the harmony just
    before and just after each frame."""
    C = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    n = C.shape[1]
    w = max(2, int(win_s * fs))
    cs = np.cumsum(np.hstack([np.zeros((12, 1)), C]), axis=1)
    nov = np.zeros(n)
    for i in range(w, n - w):
        a = (cs[:, i] - cs[:, i - w]) / w
        b = (cs[:, i + w] - cs[:, i]) / w
        nov[i] = 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return nov


def _acf(x, max_lag):
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    a = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return a / a[0] if a[0] > 0 else a


def analyze_bar_period(features):
    fs = features["sr"] / HOP
    nov = harmonic_novelty(features["chroma_smooth"], fs)
    accent = _z(features["low_env"]) + _z(features["onset_env"])
    combined = accent + 2.0 * _z(nov)

    max_lag = int(MAX_BAR_S * 1.3 * fs)
    ac_comb = _acf(combined, max_lag)
    ac_acc = _acf(accent, max_lag)
    lags = np.arange(len(ac_comb)) / fs

    pk, _ = find_peaks(ac_comb, distance=max(1, int(0.15 * fs)))
    cand = [(lags[p], ac_comb[p]) for p in pk if MIN_BAR_S <= lags[p] <= MAX_BAR_S]
    if not cand:
        return {"bar_s": None, "strength": 0.0, "candidates": []}

    top = max(s for _, s in cand)
    # fundamental: the SHORTEST peak that is nearly as strong as the best one
    strong = [(l, s) for l, s in cand if s >= 0.85 * top]
    bar_s, strength = min(strong, key=lambda t: t[0])

    return {
        "bar_s": float(bar_s),
        "strength": float(strength),
        "candidates": [(round(float(l), 2), round(float(s), 2)) for l, s in sorted(cand, key=lambda t: -t[1])[:5]],
        "_ac_accent": ac_acc,
        "_fs": fs,
    }


def _acf_at(ac, fs, lag_s, tol=0.03):
    i = lag_s * fs
    lo, hi = int(round(i * (1 - tol))), int(round(i * (1 + tol)))
    hi = min(hi, len(ac) - 1)
    return float(np.max(ac[lo:hi + 1])) if hi >= lo >= 0 else 0.0


def rescue_hypothesis(bp, tracker_bar_s):
    """
    Decide whether the tracker's bar length is inconsistent with the music.
    Returns {label, beats_per_bar, bpm, ...} or None to keep the tracker.
    """
    if not bp or bp["bar_s"] is None or bp["strength"] < MIN_STRENGTH:
        return None

    ratio = bp["bar_s"] / tracker_bar_s
    for t in (1.0, 2.0, 0.5, 4.0, 0.25):
        if abs(ratio / t - 1.0) < 0.07:
            return None   # same bar, or ordinary 1-bar vs 2-bar ambiguity

    ac, fs = bp["_ac_accent"], bp["_fs"]
    hyps = []
    for nb, label, prior in ((4, "4/4", 1.0), (3, "3/4", 0.85)):
        q = bp["bar_s"] / nb
        if not (QUARTER_MIN_S <= q <= QUARTER_MAX_S):
            continue
        # beats inside the bar should show up as periodicities of the accents
        score = np.mean([_acf_at(ac, fs, k * q) for k in range(1, nb)]) * prior
        hyps.append({"label": label, "beats_per_bar": nb, "bpm": 60.0 / q, "score": float(score)})
    if not hyps:
        return None

    hyps.sort(key=lambda h: h["score"], reverse=True)
    best = hyps[0]
    total = sum(max(h["score"], 1e-6) for h in hyps)
    best["share"] = best["score"] / total
    best["alternatives"] = hyps[1:]
    best["acf_bar_s"] = bp["bar_s"]
    best["tracker_bar_s"] = tracker_bar_s
    best["acf_strength"] = bp["strength"]
    return best
