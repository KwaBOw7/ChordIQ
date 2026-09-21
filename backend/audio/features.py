"""
Shared audio features.

Everything expensive (HPSS, CQT chroma, MFCC, onset strength) is computed
ONCE here and reused by the key, beat, chord and section modules.
"""

import numpy as np
import librosa

HOP = 512


def extract_features(y, sr):
    """Return a dict of frame-level features (hop = HOP samples)."""

    # Split harmonic (chords, melody) from percussive (drums).
    y_harm, y_perc = librosa.effects.hpss(y)

    # Songs are often slightly off A440 (tape speed, live tuning).
    tuning = float(librosa.estimate_tuning(y=y_harm, sr=sr))

    chroma = librosa.feature.chroma_cqt(
        y=y_harm,
        sr=sr,
        hop_length=HOP,
        tuning=tuning,
        bins_per_octave=36,
        n_chroma=12,
    )

    # Bass chroma: only the lowest octaves. Helps root/inversion decisions
    # and is a strong downbeat cue.
    bass = librosa.feature.chroma_cqt(
        y=y_harm,
        sr=sr,
        hop_length=HOP,
        tuning=tuning,
        fmin=librosa.note_to_hz("C1"),
        n_octaves=3,
        bins_per_octave=36,
        n_chroma=12,
    )

    # Median filtering along time suppresses passing notes / ad-libs.
    chroma_smooth = librosa.decompose.nn_filter(
        chroma,
        aggregate=np.median,
        metric="cosine",
        width=15,
    )
    chroma_smooth = np.minimum(chroma, chroma_smooth)

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP,
        aggregate=np.median,
    )

    # Low-frequency onset strength (kick / bass) -> strong beat-1 cue.
    S = np.abs(librosa.stft(y_perc, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[freqs < 180, :]
    low_env = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(low, ref=np.max),
        sr=sr,
        hop_length=HOP,
    )

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        hop_length=HOP,
        n_mfcc=13,
    )

    n = min(
        chroma.shape[1],
        chroma_smooth.shape[1],
        bass.shape[1],
        len(onset_env),
        len(low_env),
        len(rms),
        mfcc.shape[1],
    )

    return {
        "sr": sr,
        "hop": HOP,
        "tuning": tuning,
        "chroma": chroma[:, :n],
        "chroma_smooth": chroma_smooth[:, :n],
        "bass": bass[:, :n],
        "onset_env": onset_env[:n],
        "low_env": low_env[:n],
        "rms": rms[:n],
        "mfcc": mfcc[:, :n],
        "n_frames": n,
    }


def frames_to_times(frames, features):
    return librosa.frames_to_time(
        frames,
        sr=features["sr"],
        hop_length=features["hop"],
    )
