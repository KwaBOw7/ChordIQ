import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa

# 22.05 kHz is plenty for chords/key/beats and ~2x faster than 44.1 kHz.
ANALYSIS_SR = 22050


def _decode_with_ffmpeg(path, sr):
    """
    Fallback for files librosa can't open: mislabeled containers (e.g. a
    WebM/Opus or M4A file renamed .mp3, common with downloaders), video
    files, odd codecs. ffmpeg detects the real format from the content.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError(
            "This file could not be decoded. It may be a different format "
            "than its extension says. Install ffmpeg (https://ffmpeg.org) "
            "and try again, or convert the file to WAV or MP3 first."
        )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "decoded.wav"
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(sr), str(out)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out.exists():
            raise ValueError(
                "This file could not be decoded as audio: "
                + (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
            )
        return librosa.load(str(out), sr=sr, mono=True)


MIN_USABLE_SECONDS = 5.0   # shorter than this, or silent, isn't a real decode


def _looks_broken(y, sr):
    """
    librosa/mpg123 can 'succeed' on a mislabeled or corrupt file by decoding
    a few garbage frames (e.g. a video container renamed .mp3) instead of
    raising. That gets missed unless we check the RESULT, not just whether
    an exception was thrown.
    """
    if y is None or len(y) == 0:
        return True
    if len(y) / sr < MIN_USABLE_SECONDS:
        return True
    if float(np.abs(y).max()) < 1e-6:
        return True
    return False


def load_audio(file_path, sr=ANALYSIS_SR):
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path.resolve()}")

    try:
        y, sample_rate = librosa.load(str(path), sr=sr, mono=True)
        if _looks_broken(y, sample_rate):
            raise ValueError("decoded audio is empty, silent, or too short")
    except Exception:
        y, sample_rate = _decode_with_ffmpeg(path, sr)
        if _looks_broken(y, sample_rate):
            raise ValueError(
                f"This file could not be decoded as usable audio: {path}. "
                "It may be a different format than its extension says, "
                "corrupted, or not actually a music file."
            )

    return y, sample_rate
