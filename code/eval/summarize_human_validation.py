#!/usr/bin/env python3
"""Generate and validate the released QIMG-7 human-validation summaries."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from compute_image_attack_iaa import FIELDS, FIELD_LABELS, compute_field, load


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = REPO_ROOT / "evaluation/human_validation/qimg7_human_validation_packet_96_completed.csv"
DEFAULT_IAA_DIR = REPO_ROOT / "evaluation/human_validation/image_attack_iaa"
METHOD_ORDER = ("full_mm", "answer_consensus", "field_selector")


def mean(values):
    return sum(values) / len(values)


def cohen_kappa(first, second, quadratic=False):
    labels = sorted(set(first) | set(second))
    n = len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    if quadratic:
        positions = {label: i for i, label in enumerate(labels)}
        denominator = max(1, (len(labels) - 1) ** 2)

        def weight(a, b):
            return (positions[a] - positions[b]) ** 2 / denominator

        observed = sum(weight(a, b) for a, b in zip(first, second)) / n
        expected = sum(
            first_counts[a] / n * second_counts[b] / n * weight(a, b)
            for a in labels
            for b in labels
        )
        return 1.0 - observed / expected

    observed = sum(a == b for a, b in zip(first, second)) / n
    expected = sum(first_counts[x] * second_counts[x] for x in labels) / (n * n)
    return (observed - expected) / (1.0 - expected)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_answer_audit(path, output_dir):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 96:
        raise SystemExit(f"Expected 96 answer-audit rows, found {len(rows)}")

    method_rows = []
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        clean = [float(row["human_score"]) for row in selected if row["regime"].startswith("TC")]
        polluted = [float(row["human_score"]) for row in selected if row["regime"].startswith("TP")]
        agreement = mean([row["human_label"] == row["gpt4omini_label"] for row in selected])
        if len(selected) != 32 or len(clean) != 16 or len(polluted) != 16:
            raise SystemExit(f"Unexpected answer-audit coverage for {method}")
        clean_score = mean(clean)
        polluted_score = mean(polluted)
        method_rows.append({
            "method": method,
            "n": len(selected),
            "clean_text_score": f"{clean_score:.6f}",
            "polluted_text_score": f"{polluted_score:.6f}",
            "drop": f"{clean_score - polluted_score:.6f}",
            "balanced_score": f"{(clean_score + polluted_score) / 2:.6f}",
            "gpt_exact_label_agreement": f"{agreement:.6f}",
        })

    human_labels = [row["human_label"] for row in rows]
    gpt_labels = [row["gpt4omini_label"] for row in rows]
    human_scores = [float(row["human_score"]) for row in rows]
    gpt_scores = [float(row["gpt4omini_score"]) for row in rows]
    agreement_rows = [{
        "n": len(rows),
        "exact_label_agreement": f"{mean([a == b for a, b in zip(human_labels, gpt_labels)]):.6f}",
        "mean_absolute_score_difference": f"{mean([abs(a - b) for a, b in zip(human_scores, gpt_scores)]):.6f}",
        "cohen_kappa": f"{cohen_kappa(human_labels, gpt_labels):.6f}",
        "quadratic_weighted_kappa": f"{cohen_kappa(human_scores, gpt_scores, quadratic=True):.6f}",
    }]

    write_csv(
        output_dir / "qimg7_human_judge_audit_summary.csv",
        list(method_rows[0]),
        method_rows,
    )
    write_csv(
        output_dir / "qimg7_human_judge_agreement.csv",
        list(agreement_rows[0]),
        agreement_rows,
    )
    return method_rows, agreement_rows[0]


def summarize_image_iaa(first_path, second_path, output_dir):
    first = load(str(first_path))
    second = load(str(second_path))
    shared = first.index.intersection(second.index)
    if len(first) != 56 or len(second) != 56 or len(shared) != 56:
        raise SystemExit("Expected 56 matching rows in each image-attack annotation file")
    first = first.loc[shared]
    second = second.loc[shared]

    overall_rows = []
    for field in FIELDS:
        kappa, agreement, n, note, interval = compute_field(first[field], second[field])
        overall_rows.append({
            "field": field,
            "label": FIELD_LABELS[field],
            "n": n,
            "annotator_1_yes_rate": f"{(first[field].str.lower() == 'yes').mean():.6f}",
            "annotator_2_yes_rate": f"{(second[field].str.lower() == 'yes').mean():.6f}",
            "cohen_kappa": "" if kappa is None else f"{kappa:.6f}",
            "ci_low": "" if interval is None else f"{interval[0]:.6f}",
            "ci_high": "" if interval is None else f"{interval[1]:.6f}",
            "raw_agreement": f"{agreement / 100:.6f}",
            "note": note,
        })

    attack_rows = []
    for attack in sorted(first["pollution_type"].unique()):
        index = first.index[first["pollution_type"] == attack]
        if len(index) != 8:
            raise SystemExit(f"Expected 8 image-attack annotations for {attack}")
        for field in FIELDS:
            kappa, agreement, n, note, interval = compute_field(first.loc[index, field], second.loc[index, field])
            attack_rows.append({
                "pollution_type": attack,
                "field": field,
                "n": n,
                "cohen_kappa": "" if kappa is None else f"{kappa:.6f}",
                "ci_low": "" if interval is None else f"{interval[0]:.6f}",
                "ci_high": "" if interval is None else f"{interval[1]:.6f}",
                "raw_agreement": f"{agreement / 100:.6f}",
                "note": note,
            })

    write_csv(output_dir / "overall_agreement.csv", list(overall_rows[0]), overall_rows)
    write_csv(output_dir / "by_attack_agreement.csv", list(attack_rows[0]), attack_rows)
    return overall_rows, attack_rows


def validate_release(answer_rows, answer_agreement, iaa_rows, attack_rows):
    expected_methods = {
        "full_mm": (0.921875, 0.406250, 0.6640625, 0.781250),
        "answer_consensus": (0.921875, 0.375000, 0.6484375, 0.812500),
        "field_selector": (0.921875, 0.750000, 0.8359375, 0.843750),
    }
    for row in answer_rows:
        actual = tuple(float(row[key]) for key in (
            "clean_text_score", "polluted_text_score", "balanced_score", "gpt_exact_label_agreement"
        ))
        if any(abs(a - b) > 1e-6 for a, b in zip(actual, expected_methods[row["method"]])):
            raise SystemExit(f"Answer-audit summary mismatch for {row['method']}")

    expected_agreement = (0.8125, 0.0963541667, 0.6414937759, 0.7675328304)
    actual_agreement = tuple(float(answer_agreement[key]) for key in (
        "exact_label_agreement", "mean_absolute_score_difference", "cohen_kappa", "quadratic_weighted_kappa"
    ))
    if any(abs(a - b) > 1e-6 for a, b in zip(actual_agreement, expected_agreement)):
        raise SystemExit("Answer-audit agreement summary mismatch")

    expected_iaa = {
        "on_topic": (44 / 56, 49 / 56, 0.700),
        "fact_flipped": (41 / 56, 46 / 56, 0.660),
        "plausible": (37 / 56, 37 / 56, 0.778),
    }
    for row in iaa_rows:
        expected = expected_iaa[row["field"]]
        actual = (
            float(row["annotator_1_yes_rate"]),
            float(row["annotator_2_yes_rate"]),
            round(float(row["cohen_kappa"]), 3),
        )
        if abs(actual[0] - expected[0]) > 1e-6 or abs(actual[1] - expected[1]) > 1e-6 or actual[2] != expected[2]:
            raise SystemExit(f"Image-attack agreement mismatch for {row['field']}")
    if len(attack_rows) != 21:
        raise SystemExit("Expected 21 per-attack agreement rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--annotator-1", type=Path, default=DEFAULT_IAA_DIR / "annotator_1.csv")
    parser.add_argument("--annotator-2", type=Path, default=DEFAULT_IAA_DIR / "annotator_2.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "evaluation/human_validation")
    args = parser.parse_args()

    answer_rows, answer_agreement = summarize_answer_audit(args.answer_audit, args.output_dir)
    iaa_rows, attack_rows = summarize_image_iaa(
        args.annotator_1,
        args.annotator_2,
        args.output_dir / "image_attack_iaa",
    )
    validate_release(answer_rows, answer_agreement, iaa_rows, attack_rows)
    print("Human-validation summaries regenerated and validated.")


if __name__ == "__main__":
    main()
