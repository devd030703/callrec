#!/usr/bin/env python3
"""Score known-reference ASR results without loading audio or models."""

import argparse
import json
import unicodedata
from collections import defaultdict


def normalise(value):
    value = unicodedata.normalize("NFC", value).casefold()
    return " ".join(
        "".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in value).split()
    )


def distance(left, right):
    previous = list(range(len(right) + 1))
    for index, item in enumerate(left, 1):
        current = [index]
        for column, other in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (item != other))
            )
        previous = current
    return previous[-1]


def score(rows):
    groups = defaultdict(
        lambda: {
            "words": 0,
            "characters": 0,
            "base_words": 0,
            "candidate_words": 0,
            "base_chars": 0,
            "candidate_chars": 0,
        }
    )
    failures = []
    if not rows:
        raise ValueError("Benchmark must contain at least one case.")
    seen = set()
    for row in rows:
        ident = row["id"]
        if ident in seen:
            raise ValueError(f"Duplicate benchmark id: {ident}")
        seen.add(ident)
        reference = normalise(row["reference"])
        base = normalise(row["baseline"])
        candidate = normalise(row["candidate"])
        if not reference:
            raise ValueError(f"Empty reference: {ident}")
        group = groups[row["language"]]
        group["words"] += len(reference.split())
        group["characters"] += len(reference)
        group["base_words"] += distance(reference.split(), base.split())
        group["candidate_words"] += distance(reference.split(), candidate.split())
        group["base_chars"] += distance(reference, base)
        group["candidate_chars"] += distance(reference, candidate)
        for text in row.get("required_phrases", []):
            if normalise(text) not in candidate:
                failures.append({"id": ident, "reason": "required_phrase_missing", "phrase": text})
        for text in row.get("forbidden_phrases", []):
            if normalise(text) in candidate:
                failures.append({"id": ident, "reason": "forbidden_phrase_present", "phrase": text})
    result = {}
    totals = {
        k: sum(g[k] for g in groups.values())
        for k in (
            "words",
            "characters",
            "base_words",
            "candidate_words",
            "base_chars",
            "candidate_chars",
        )
    }
    for language, group in sorted(groups.items()):
        result[language] = {
            "words": group["words"],
            "baseline_wer": group["base_words"] / group["words"],
            "candidate_wer": group["candidate_words"] / group["words"],
            "baseline_cer": group["base_chars"] / group["characters"],
            "candidate_cer": group["candidate_chars"] / group["characters"],
        }
        if result[language]["candidate_wer"] > result[language]["baseline_wer"] + 0.01:
            failures.append({"language": language, "reason": "wer_regression_over_one_point"})
    baseline = totals["base_words"] / totals["words"]
    candidate = totals["candidate_words"] / totals["words"]
    if baseline == 0 or candidate > baseline * 0.9:
        failures.append({"reason": "less_than_ten_percent_relative_wer_improvement"})
    return {
        "benchmark_only": True,
        "quality_verified": False,
        "cases": len(rows),
        "baseline_wer": baseline,
        "candidate_wer": candidate,
        "by_language": result,
        "failures": failures,
        "promotion_gate_passed": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="JSON array of reference, baseline, and candidate strings")
    args = parser.parse_args()
    with open(args.results, encoding="utf-8") as f:
        report = score(json.load(f))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
