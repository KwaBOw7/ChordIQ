from audio.harmony import add_nashville_numbers, quality_suffix


def chord_label(chord):
    return f"{chord['root']}{quality_suffix(chord['quality'])}"


def build_chord_sequence(chords):
    if not chords:
        return []

    sequence = []
    current = None

    for chord in chords:
        label = chord_label(chord)

        item = {
            "chord": label,
            "root": chord["root"],
            "quality": chord["quality"],
            "start": float(chord["start"]),
            "end": float(chord["end"]),
            "duration": float(
                chord["end"] - chord["start"]
            ),
            "confidence": float(
                chord.get("confidence", 0.0)
            ),
            "bar": chord.get("bar"),
            "key_tonic": chord.get("key_tonic"),
            "key_mode": chord.get("key_mode"),
        }

        if current is None:
            current = item
            continue

        if label == current["chord"]:
            current["end"] = item["end"]

            current["duration"] = (
                current["end"] - current["start"]
            )

            current["confidence"] = (
                current["confidence"]
                + item["confidence"]
            ) / 2.0

        else:
            sequence.append(current)
            current = item

    if current is not None:
        sequence.append(current)

    return sequence


def find_repeated_patterns(
    sequence,
    min_length=4,
    max_length=8,
    min_occurrences=2,
):
    """
    Find repeated chord progressions.

    Repetition is based on the actual detected chords,
    not merely their Nashville numbers. This prevents
    different harmonic spellings from being incorrectly
    merged.
    """

    labels = [
        item["chord"]
        for item in sequence
    ]

    n = len(labels)

    if n < min_length:
        return []

    patterns = []

    for length in range(
        min_length,
        min(max_length, n) + 1,
    ):
        occurrences = {}

        for start in range(
            n - length + 1
        ):
            pattern = tuple(
                labels[start:start + length]
            )

            occurrences.setdefault(
                pattern,
                []
            ).append(start)

        for pattern, positions in occurrences.items():

            if len(positions) < min_occurrences:
                continue

            locations = []

            for position in positions:
                first = sequence[position]
                last = sequence[
                    position + length - 1
                ]

                items = sequence[
                    position:
                    position + length
                ]

                locations.append({
                    "start": first["start"],
                    "end": last["end"],

                    "chords": [
                        item["chord"]
                        for item in items
                    ],

                    "nashville": [
                        item.get(
                            "nashville",
                            item.get("number")
                        )
                        for item in items
                    ],
                })

            patterns.append({
                "pattern": list(pattern),

                "nashville": [
                    sequence[
                        positions[0] + i
                    ].get(
                        "nashville",
                        sequence[
                            positions[0] + i
                        ].get("number")
                    )
                    for i in range(length)
                ],

                "length": length,
                "occurrences": len(locations),
                "locations": locations,
            })

    return patterns


def _is_subsequence_pattern(shorter, longer):
    shorter = tuple(shorter)
    longer = tuple(longer)

    if len(shorter) >= len(longer):
        return False

    return any(
        longer[i:i + len(shorter)] == shorter
        for i in range(
            len(longer) - len(shorter) + 1
        )
    )


def remove_redundant_patterns(patterns):
    """
    Prefer meaningful longer repeated patterns
    over trivial shorter patterns.
    """

    ordered = sorted(
        patterns,
        key=lambda p: (
            p["length"],
            p["occurrences"],
        ),
        reverse=True,
    )

    selected = []

    for candidate in ordered:

        if any(
            existing["length"]
            > candidate["length"]
            and _is_subsequence_pattern(
                candidate["pattern"],
                existing["pattern"],
            )
            for existing in selected
        ):
            continue

        selected.append(candidate)

    return sorted(
        selected,
        key=lambda p: (
            -p["occurrences"],
            -p["length"],
        ),
    )


def analyze_progression(
    chords,
    key,
    mode="major",
):
    sequence = build_chord_sequence(chords)

    sequence = add_nashville_numbers(
        sequence,
        key,
        mode,
    )

    patterns = find_repeated_patterns(
        sequence
    )

    patterns = remove_redundant_patterns(
        patterns
    )

    return {
        "sequence": sequence,
        "patterns": patterns,
    }