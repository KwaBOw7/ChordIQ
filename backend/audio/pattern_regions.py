def _number_name(item):
    if isinstance(item, str):
        return item

    return (
        item.get("number")
        or item.get("chord", "")
    )


def _chord_name(item):
    if isinstance(item, str):
        return item

    return item.get(
        "chord",
        "",
    )


def _make_signature(items):
    return tuple(
        _number_name(item)
        for item in items
    )


def _same_pattern(a, b):
    a_values = [
        _number_name(x)
        for x in a
    ]

    b_values = [
        _number_name(x)
        for x in b
    ]

    if len(a_values) != len(b_values):
        return False

    if a_values == b_values:
        return True

    return any(
        a_values
        == b_values[i:]
        + b_values[:i]
        for i in range(
            len(b_values)
        )
    )


def detect_alternating_vamp(
    sequence,
    min_length=4,
):
    regions = []

    i = 0

    while (
        i + min_length
        <= len(sequence)
    ):

        first = _number_name(
            sequence[i]
        )

        second = _number_name(
            sequence[i + 1]
        )

        if first == second:
            i += 1
            continue

        j = i + 2

        while j < len(sequence):

            expected = (
                first
                if (j - i) % 2 == 0
                else second
            )

            if (
                _number_name(
                    sequence[j]
                )
                != expected
            ):
                break

            j += 1

        if j - i >= min_length:

            regions.append({
                "type": "alternating",

                "pattern": [
                    first,
                    second,
                ],

                "start": sequence[i]["start"],

                "end": sequence[
                    j - 1
                ]["end"],

                "length": j - i,
            })

            i = j

        else:
            i += 1

    return regions


def detect_repeated_regions(
    sequence,
    pattern_length=4,
    min_occurrences=2,
):
    if len(sequence) < pattern_length:
        return []

    candidates = {}

    for i in range(
        len(sequence)
        - pattern_length
        + 1
    ):

        signature = _make_signature(
            sequence[
                i:
                i + pattern_length
            ]
        )

        candidates.setdefault(
            signature,
            [],
        ).append(i)

    repeated = []

    for signature, indexes in candidates.items():

        if len(indexes) < min_occurrences:
            continue

        occurrences = []

        for index in indexes:

            items = sequence[
                index:
                index + pattern_length
            ]

            occurrences.append({
                "start": items[0]["start"],
                "end": items[-1]["end"],

                "pattern": [
                    _number_name(x)
                    for x in items
                ],

                "chords": [
                    _chord_name(x)
                    for x in items
                ],
            })

        repeated.append({
            "signature": signature,
            "occurrences": occurrences,
        })

    return repeated


def merge_similar_patterns(patterns):
    merged = []

    for pattern in patterns:

        representative = (
            pattern["occurrences"][0]
            ["pattern"]
        )

        matched = False

        for existing in merged:

            existing_rep = (
                existing["occurrences"][0]
                ["pattern"]
            )

            if _same_pattern(
                representative,
                existing_rep,
            ):

                existing[
                    "occurrences"
                ].extend(
                    pattern[
                        "occurrences"
                    ]
                )

                matched = True
                break

        if not matched:

            merged.append({
                "signature": pattern[
                    "signature"
                ],

                "occurrences": list(
                    pattern[
                        "occurrences"
                    ]
                ),
            })

    return merged


def analyze_pattern_regions(
    sequence
):
    if not sequence:
        return {
            "regions": [],
            "vamps": [],
        }

    repeated = detect_repeated_regions(
        sequence
    )

    repeated = merge_similar_patterns(
        repeated
    )

    vamps = detect_alternating_vamp(
        sequence
    )

    return {
        "regions": repeated,
        "vamps": vamps,
    }