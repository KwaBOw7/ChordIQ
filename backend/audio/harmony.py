import re

NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

# chord-name suffix  <->  quality
SUFFIX_TO_QUALITY = {
    "": "major",
    "m": "minor",
    "7": "dominant7",
    "maj7": "major7",
    "m7": "minor7",
    "dim": "diminished",
}
QUALITY_TO_SUFFIX = {v: k for k, v in SUFFIX_TO_QUALITY.items()}

_CHORD_RE = re.compile(r"^([A-G](?:#|b)?)(.*)$")

# Semitones above the tonic -> Nashville degree
_DEGREE = {
    0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
    6: "#4", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7",
}


def quality_suffix(quality):
    return QUALITY_TO_SUFFIX.get(quality, "")


def split_chord(chord):
    """'Abm7' -> ('Ab', 'minor7')"""
    m = _CHORD_RE.match(chord or "")
    if not m:
        return chord, "major"
    root, suffix = m.group(1), m.group(2)
    return root, SUFFIX_TO_QUALITY.get(suffix, "major")


def chord_to_nashville(chord, key, mode="major"):
    """Nashville number of `chord` relative to `key` ('Ab' -> '1', 'Fm7' -> '6m7')."""
    root, quality = split_chord(chord)
    if root not in NOTE_TO_SEMITONE or key not in NOTE_TO_SEMITONE:
        return None
    interval = (NOTE_TO_SEMITONE[root] - NOTE_TO_SEMITONE[key]) % 12
    suffix = quality_suffix(quality)
    if suffix == "7":
        suffix = "\u2077"   # 5 + superscript 7, never "57"
    return f"{_DEGREE[interval]}{suffix}"


_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def add_nashville_numbers(sequence, key, mode="major"):
    """
    Add a Nashville number to every chord. If an item carries its own local
    key (key_tonic, from the key-change timeline) that key is used, so a
    song that modulates up a semitone keeps reading 1-4-5.
    """
    result = []
    for item in sequence:
        updated = dict(item)
        k, m = key, mode
        if item.get("key_tonic") is not None:
            k = _SHARP[int(item["key_tonic"]) % 12]
            m = item.get("key_mode", mode)
        updated["nashville"] = chord_to_nashville(item.get("chord"), k, m)
        updated["number"] = updated["nashville"]   # compatibility alias
        result.append(updated)
    return result
