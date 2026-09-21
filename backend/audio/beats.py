"""
Beat tracking, tempo, time signature and bar (downbeat) detection.

Pipeline
1. Beat-track the onset envelope, fold tempo into a musical range.
2. Beat-synchronous accents: low-frequency onset (kick), full-band onset,
   and harmonic novelty (chord changes).
3. Time signature = which bar length (2/3/4/6 beats) the accents repeat at,
   plus a triplet-feel test to tell 4/4 from 12/8 and 3/4 from 9/8.
4. Downbeat phase = which beat inside the bar carries the strongest
   kick + chord-change evidence.
"""

import numpy as np
import librosa

from audio.features import HOP
from audio.barperiod import analyze_bar_period, rescue_hypothesis

TEMPO_LOW = 50.0
TEMPO_HIGH = 160.0


# ------------------------------------------------------------------ helpers
def _z(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def _env_at(env, frame, radius=1):
    lo = max(0, int(frame) - radius)
    hi = min(len(env), int(frame) + radius + 1)
    if hi <= lo:
        return 0.0
    return float(np.max(env[lo:hi]))


def _beat_accents(env, beat_frames, radius=2):
    return np.array([_env_at(env, f, radius) for f in beat_frames])


def _beat_chroma(chroma, beat_frames):
    """Mean chroma inside each beat interval (len = n_beats)."""
    out = []
    n = chroma.shape[1]
    edges = list(beat_frames) + [min(n, beat_frames[-1] + (beat_frames[-1] - beat_frames[-2]))]
    for a, b in zip(edges[:-1], edges[1:]):
        a = int(max(0, min(n - 1, a)))
        b = int(max(a + 1, min(n, b)))
        seg = chroma[:, a:b]
        out.append(np.median(seg, axis=1) if seg.shape[1] > 2 else seg.mean(axis=1))
    return np.array(out).T  # (12, n_beats)


def _novelty(beat_chroma):
    """1 - cosine(previous beat, this beat): high where the harmony changes."""
    X = beat_chroma / (np.linalg.norm(beat_chroma, axis=0, keepdims=True) + 1e-9)
    nov = np.zeros(X.shape[1])
    nov[1:] = 1.0 - np.sum(X[:, 1:] * X[:, :-1], axis=0)
    return nov


def _acf(x, max_lag=16):
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.dot(x, x)
    if denom <= 0:
        return np.zeros(max_lag + 1)
    out = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        if lag < len(x):
            out[lag] = np.dot(x[:-lag], x[lag:]) / denom
    out[0] = 1.0
    return out


def _peak(ac, lag):
    """Height of the autocorrelation peak at `lag` above its neighbours."""
    if lag + 1 >= len(ac):
        return 0.0
    return max(0.0, ac[lag] - 0.5 * (ac[lag - 1] + ac[lag + 1]))


def _period_strength(ac, b):
    """Evidence that the pattern repeats every b beats (also checks 2b)."""
    s = _peak(ac, b)
    if 2 * b + 1 < len(ac):
        s = 0.65 * s + 0.35 * _peak(ac, 2 * b)
    return s


def _onset_peaks(env_low, env_full):
    from scipy.signal import find_peaks
    comb = _z(env_low) + _z(env_full)
    peaks, _ = find_peaks(comb, height=comb.mean() + 0.8 * comb.std(), distance=5)
    if len(peaks) == 0:
        return peaks, np.zeros(0)
    # keep only the strongest 30% of hits (kick / snare / chord attacks):
    # hi-hats and ghost notes happen on every subdivision and prove nothing
    heights = comb[peaks]
    keep = heights >= np.quantile(heights, 0.70)
    peaks = peaks[keep]
    return peaks, np.ones(len(peaks))


def _coverage(onsets, weights, grid_frames, period):
    """Share of onset STRENGTH that lands on the grid (+/-12% of a period)."""
    if len(onsets) == 0 or len(grid_frames) == 0:
        return 0.0
    tol = max(1.0, 0.12 * period)
    grid = np.asarray(grid_frames, dtype=float)
    idx = np.searchsorted(grid, onsets)
    lo = grid[np.clip(idx - 1, 0, len(grid) - 1)]
    hi = grid[np.clip(idx, 0, len(grid) - 1)]
    d = np.minimum(np.abs(onsets - lo), np.abs(onsets - hi))
    total = float(np.sum(weights))
    return float(np.sum(weights[d <= tol]) / total) if total > 0 else 0.0


def refine_pulse(beat_frames, env_low, env_full):
    """
    In compound meters (6/8, 12/8) the tracker often locks onto a pulse of
    2 eighth notes, which is not the musical beat (3 eighths). Test whether a
    grid 1.5x wider explains the strong onsets clearly better.
    Returns (beat_frames, changed).
    """
    onsets, weights = _onset_peaks(env_low, env_full)
    period = float(np.median(np.diff(beat_frames)))
    base = _coverage(onsets, weights, beat_frames, period)

    idx = np.arange(len(beat_frames))
    best_cov, best_frames = base, None
    for phase in (0.0, 0.5, 1.0):
        pos = np.arange(phase, len(beat_frames) - 1, 1.5)
        if len(pos) < 8:
            continue
        frames = np.interp(pos, idx, beat_frames)
        cov = _coverage(onsets, weights, frames, 1.5 * period)
        if cov > best_cov:
            best_cov, best_frames = cov, frames

    # the wider grid must be CLEARLY better (it starts at a disadvantage)
    if best_frames is not None and best_cov >= base + 0.10:
        return np.round(best_frames).astype(int), True
    return beat_frames, False


def _extend_grid(beat_frames, n_frames):
    beat_frames = np.asarray(beat_frames, dtype=int)
    iv = float(np.median(np.diff(beat_frames)))
    first, last = float(beat_frames[0]), float(beat_frames[-1])

    pre = []
    k = 1
    while first - k * iv >= -0.25 * iv:
        pre.append(int(round(max(0.0, first - k * iv))))
        k += 1
    pre = pre[::-1]

    post = []
    k = 1
    while last + k * iv <= n_frames - 1 - 0.25 * iv:
        post.append(int(round(last + k * iv)))
        k += 1

    return np.array(pre + list(beat_frames) + post, dtype=int)


# ------------------------------------------------------------ triplet feel
def triplet_feel(env, beat_frames):
    """
    0..1. Compares onset energy at the 1/2 point of each beat (straight
    8ths) with the 1/3 and 2/3 points (triplets / shuffle / 12-8).
    """
    tri, dup = [], []
    for a, b in zip(beat_frames[:-1], beat_frames[1:]):
        span = b - a
        if span < 6:
            continue
        dup.append(_env_at(env, a + 0.5 * span, 1))
        tri.append(0.5 * (_env_at(env, a + span / 3.0, 1) + _env_at(env, a + 2 * span / 3.0, 1)))
    if not tri:
        return 0.0
    tri_m, dup_m = float(np.mean(tri)), float(np.mean(dup))
    # Straight music has energy at 1/2 and 1/4; triplet music at 1/3 and 2/3.
    return tri_m / (tri_m + dup_m + 1e-9)


# ------------------------------------------------------------- time signature
METER_INFO = {
    "4/4": 4,
    "3/4": 3,
    "2/4": 2,
    "12/8": 4,
    "6/8": 2,
    "9/8": 3,
}


def estimate_meter(acc, nov, tri):
    """
    acc, nov: beat-synchronous accent / harmonic-novelty series.
    tri: triplet-feel value.
    Returns (label, beats_per_bar, confidence, ranked list).
    """
    ac_a = _acf(acc)
    ac_n = _acf(nov)

    def s(b):
        return 0.6 * _period_strength(ac_a, b) + 0.4 * _period_strength(ac_n, b)

    s2, s3, s4, s6 = s(2), s(3), s(4), s(6)

    compound = 1.0 if tri >= 0.52 else 0.0
    if 0.45 <= tri < 0.52:
        compound = (tri - 0.45) / 0.07  # soft zone
    straight = 1.0 - compound

    scores = {
        "4/4": s4 * straight * 1.00,
        "3/4": s3 * straight * 0.90,
        "2/4": s2 * straight * 0.45,
        "12/8": s4 * compound * 0.90,
        "9/8": s3 * compound * 0.55,
        "6/8": max(s2, 0.8 * s6) * compound * 0.85,
    }

    # If the tracker follows the eighth-note pulse of a 6/8 song, the accent
    # pattern repeats every 6 beats and 3 beats instead of 2.
    if compound and s6 > max(s2, s4):
        scores["6/8"] = max(scores["6/8"], s6 * compound)

    # Straight 4/4 whose accent repeats every 8 (2 bars) is still 4/4.
    total = sum(scores.values()) + 1e-9
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    if best_score <= 0:
        best = "4/4"
    confidence = best_score / total if best_score > 0 else 0.0
    return best, METER_INFO[best], float(confidence), [
        {"label": k, "score": float(v / total)} for k, v in ranked[:3]
    ]


# ------------------------------------------------------------------ bars
def find_downbeat_phase(low_z, full_z, nov_z, bpb):
    best_phase, best_score = 0, -1e9
    for p in range(bpb):
        idx = np.arange(p, len(low_z), bpb)
        if len(idx) < 2:
            continue
        score = (
            1.0 * np.mean(low_z[idx])
            + 0.7 * np.mean(nov_z[idx])
            + 0.3 * np.mean(full_z[idx])
        )
        if score > best_score:
            best_phase, best_score = p, score
    return best_phase


def build_bars(beat_times, phase, bpb):
    bars = []
    n = len(beat_times)
    i = phase
    index = 0
    while i < n - 1:
        j = min(i + bpb, n - 1)
        bars.append({
            "index": index,
            "start": float(beat_times[i]),
            "end": float(beat_times[j]),
            "beat_start": int(i),
            "beat_end": int(j),
        })
        index += 1
        i += bpb
    return bars


# ----------------------------------------------------------------- main API
def analyze_beats(y, sample_rate, features, forced_meter=None, forced_bpm=None, allow_rescue=True):
    env = features["onset_env"]

    track_kwargs = {"start_bpm": 100}
    if forced_bpm:
        # user knows the real beat (e.g. 70 BPM): track around THAT tempo
        track_kwargs = {"bpm": float(forced_bpm)}

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=env,
        sr=sample_rate,
        hop_length=HOP,
        tightness=100,
        trim=False,
        **track_kwargs,
    )
    beat_frames = np.asarray(beat_frames, dtype=int)
    if len(beat_frames) < 8:
        raise ValueError("Could not find a steady beat in this audio.")

    low_env = features["low_env"]
    acc_low = _beat_accents(low_env, beat_frames)
    acc_full = _beat_accents(env, beat_frames)

    def median_bpm(frames):
        iv = np.diff(frames) * HOP / sample_rate
        return 60.0 / float(np.median(iv))

    if forced_bpm:
        regridded = False
    else:
        beat_frames, regridded = refine_pulse(beat_frames, low_env, env)
    if regridded:
        acc_low = _beat_accents(low_env, beat_frames)
        acc_full = _beat_accents(env, beat_frames)

    bpm = median_bpm(beat_frames)

    # Fold tempo into a musical range and re-grid the beats to match.
    fold = "none"
    if regridded or forced_bpm:
        pass    # compound pulse or user-given tempo: keep the grid as is
    elif bpm >= TEMPO_HIGH:
        fold = "halved"
        ph = 0 if (acc_low[0::2].mean() + acc_full[0::2].mean()) >= (
            acc_low[1::2].mean() + acc_full[1::2].mean()
        ) else 1
        beat_frames = beat_frames[ph::2]
    elif bpm < TEMPO_LOW:
        fold = "doubled"
        mids = ((beat_frames[:-1] + beat_frames[1:]) // 2)
        merged = np.empty(len(beat_frames) + len(mids), dtype=int)
        merged[0::2] = beat_frames
        merged[1::2] = mids
        beat_frames = merged

    bpm = median_bpm(beat_frames)

    # The tracker usually skips the very first beat(s) and stops early.
    # Extend the grid back to the start and forward to the end so bar 1
    # really is bar 1.
    beat_frames = _extend_grid(beat_frames, features["n_frames"])
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=HOP)
    intervals = np.diff(beat_times)

    # Recompute accents on the final beat grid
    acc_low = _beat_accents(low_env, beat_frames)
    acc_full = _beat_accents(env, beat_frames)
    bchroma = _beat_chroma(features["chroma_smooth"], beat_frames)
    bbass = _beat_chroma(features["bass"], beat_frames)
    nov = 0.5 * _novelty(bchroma) + 0.5 * _novelty(bbass)

    low_z, full_z, nov_z = _z(acc_low), _z(acc_full), _z(nov)
    acc = 1.0 * low_z + 0.5 * full_z

    tri = triplet_feel(env, beat_frames)
    label, bpb, confidence, ranked = estimate_meter(acc, nov_z, tri)
    forced = False
    if forced_meter in METER_INFO:
        label, bpb, forced = forced_meter, METER_INFO[forced_meter], True
    phase = find_downbeat_phase(low_z, full_z, nov_z, bpb)
    bars = build_bars(beat_times, phase, bpb)

    stability = float(np.std(intervals) / np.mean(intervals)) if len(intervals) else 0.0

    result = {
        "tempo": float(bpm),
        "measured_bpm": float(bpm),
        "tempo_half": float(bpm / 2.0),
        "tempo_double": float(bpm * 2.0),
        "tempo_fold": fold,
        "tempo_user_forced": bool(forced_bpm),
        "compound_pulse_detected": bool(regridded),
        "tempo_stability": stability,
        "beat_times": beat_times.tolist(),
        "beats": beat_times.tolist(),
        "beat_frames": beat_frames.tolist(),
        "beat_intervals": intervals.tolist(),
        "time_signature": {
            "label": label,
            "beats_per_bar": int(bpb),
            "confidence": confidence,
            "alternatives": ranked,
            "triplet_feel": float(tri),
            "user_forced": forced,
        },
        "downbeat_phase": int(phase),
        "bars": bars,
        "beat_chroma": bchroma,   # (12, n_beats), used by the chord module
        "beat_bass": bbass,
    }

    result["bar_rescue"] = None

    # Second opinion on the bar length (skipped when the user gave the tempo)
    if allow_rescue and not forced_bpm and len(bars) > 2:
        tracker_bar = float(np.median([b["end"] - b["start"] for b in bars]))
        hyp = rescue_hypothesis(analyze_bar_period(features), tracker_bar)
        if hyp is not None:
            fixed = analyze_beats(
                y, sample_rate, features,
                forced_meter=hyp["label"], forced_bpm=hyp["bpm"],
                allow_rescue=False,
            )
            fixed["tempo_user_forced"] = False
            fixed["time_signature"]["user_forced"] = False
            fixed["time_signature"]["confidence"] = float(hyp["share"])
            fixed["bar_rescue"] = {
                "tracker_bar_s": hyp["tracker_bar_s"],
                "music_bar_s": hyp["acf_bar_s"],
                "tracker_tempo": result["tempo"],
                "tracker_meter": result["time_signature"]["label"],
                "strength": hyp["acf_strength"],
            }
            return fixed

    return result

