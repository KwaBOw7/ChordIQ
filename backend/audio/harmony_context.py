"""
Fast harmonic stabilization.

Purpose:
- Clean obvious one-frame chord glitches.
- Preserve genuine short chord changes.
- Preserve major/minor distinctions.
- Preserve diminished and chromatic chords.
- Avoid expensive all-to-all pattern comparisons.

This module is intentionally O(n).
"""

from collections import Counter


def _get_chord(item):
    """Safely extract a chord name from different possible sequence formats."""

    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        return (
            item.get("chord")
            or item.get("name")
            or item.get("label")
            or item.get("chord_name")
            or ""
        )

    # Support tuple/list formats such as:
    # ("G#", start, end)
    if isinstance(item, (tuple, list)):
        if len(item) > 0:
            return str(item[0])

    return str(item)


def _get_start(item):
    if isinstance(item, dict):
        return item.get("start", item.get("start_time", 0.0))

    if isinstance(item, (tuple, list)) and len(item) > 1:
        return item[1]

    return 0.0


def _get_end(item):
    if isinstance(item, dict):
        return item.get("end", item.get("end_time", 0.0))

    if isinstance(item, (tuple, list)) and len(item) > 2:
        return item[2]

    return 0.0


def _duration(item):
    try:
        return float(_get_end(item)) - float(_get_start(item))
    except (TypeError, ValueError):
        return 0.0


def _copy_with_chord(item, chord):
    """
    Return a copy of an item with its chord replaced.

    This allows the stabilizer to work with the dictionaries produced
    by the chord analyzer without destroying timing information.
    """

    if isinstance(item, dict):
        result = dict(item)

        if "chord" in result:
            result["chord"] = chord
        elif "name" in result:
            result["name"] = chord
        elif "label" in result:
            result["label"] = chord
        elif "chord_name" in result:
            result["chord_name"] = chord
        else:
            result["chord"] = chord

        return result

    if isinstance(item, (tuple, list)):
        result = list(item)

        if result:
            result[0] = chord

        return result

    return chord


def _same_chord(a, b):
    return _get_chord(a) == _get_chord(b)


def _is_short_glitch(item, minimum_duration=0.65):
    """
    A chord this short is suspicious, but we do NOT automatically remove it.

    It is only removed when surrounded by the same chord.
    """

    return _duration(item) > 0 and _duration(item) < minimum_duration


def stabilize_chords(
    sequence,
    minimum_glitch_duration=0.65,
):
    """
    Fast single-pass chord stabilizer.

    Example:

        G# -> G#dim -> G#

    where G#dim lasts only 0.4s becomes:

        G# -> G#

    But:

        G# -> G#m -> G#

    is preserved when G#m lasts long enough.

    Returns a new sequence.
    """

    if not sequence:
        return []

    sequence = list(sequence)

    # Work on copies so the original analysis is not mutated.
    result = [_copy_with_chord(x, _get_chord(x)) for x in sequence]

    if len(result) < 3:
        return result

    i = 1

    while i < len(result) - 1:

        previous = result[i - 1]
        current = result[i]
        following = result[i + 1]

        current_chord = _get_chord(current)
        previous_chord = _get_chord(previous)
        following_chord = _get_chord(following)

        # Only remove a short chord when both neighboring chords agree.
        #
        # Example:
        # G# -> G#dim -> G#
        #
        # This is a strong indication that G#dim was a detection artifact.
        if (
            previous_chord == following_chord
            and current_chord != previous_chord
            and _is_short_glitch(
                current,
                minimum_glitch_duration,
            )
        ):
            result[i] = _copy_with_chord(
                current,
                previous_chord,
            )

        i += 1

    # Merge consecutive identical chords.
    merged = []

    for item in result:
        chord = _get_chord(item)

        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]

        if _get_chord(previous) == chord:

            # Preserve the first item's structure and extend its end time.
            if isinstance(previous, dict) and isinstance(item, dict):

                combined = dict(previous)

                if "end" in combined and "end" in item:
                    combined["end"] = item["end"]

                if "end_time" in combined and "end_time" in item:
                    combined["end_time"] = item["end_time"]

                merged[-1] = combined

            elif isinstance(previous, list) and isinstance(item, list):
                combined = list(previous)

                if len(combined) > 2 and len(item) > 2:
                    combined[2] = item[2]

                merged[-1] = combined

            else:
                merged[-1] = previous

        else:
            merged.append(item)

    return merged


def classify_harmonic_role(chord, key_name):
    """
    Assign a simple harmonic role.

    This is intentionally lightweight. It is not used to alter the
    detected chord; it only describes its relationship to the key.
    """

    if not chord:
        return "unknown"

    chord_clean = chord.replace("♭", "b").replace("♯", "#")

    # Normalize the tonic spelling enough for the common output of the
    # current key detector.
    tonic = key_name.split()[0] if key_name else ""

    if chord_clean == tonic:
        return "tonic"

    # Common relationships for the current Nashville-number system.
    roles = {
        "I": "tonic",
        "ii": "predominant",
        "iii": "tonic substitute",
        "IV": "predominant",
        "V": "dominant",
        "vi": "tonic substitute",
        "vii°": "dominant substitute",
    }

    # Role assignment is completed later by the progression analyzer
    # when Nashville numbers are available.
    return roles.get(chord_clean, "chromatic")


def analyze_harmonic_context(sequence, key_name=None):
    """
    Main entry point.

    Returns:
        {
            "sequence": stabilized sequence,
            "statistics": {...}
        }
    """

    if sequence is None:
        sequence = []

    stabilized = stabilize_chords(sequence)

    chord_names = [
        _get_chord(item)
        for item in stabilized
        if _get_chord(item)
    ]

    counts = Counter(chord_names)

    statistics = {
        "original_count": len(sequence),
        "stabilized_count": len(stabilized),
        "unique_chords": len(counts),
        "chord_counts": dict(counts),
    }

    if key_name:
        statistics["key"] = key_name

    return {
        "sequence": stabilized,
        "statistics": statistics,
    }