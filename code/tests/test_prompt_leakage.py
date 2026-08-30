#!/usr/bin/env python3
"""Fail if any prompt builder can put benchmark construction data in a prompt.

Runs every released benchmark row through every prompt builder and scans the
serialized result for the construction fields (`regime`, `text_status`,
`image_status`, `image_pollution_type`, and the `image_evidence.metadata`
block). The paper states these are stripped before answer, router, and judge
prompts; this is the check that keeps that true.

Run directly (`python code/tests/test_prompt_leakage.py`) or under pytest.
Needs no API key and makes no network calls.
"""

import importlib.util
import json
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "code"))

from common.prompt_fields import (  # noqa: E402
    ALLOW_LISTS, find_leakage, model_facing_evidence,
)

DATASETS = [("fava", 20), ("biography", 20), ("alpacafact", 20), ("longfact", 50)]


def _stub_openai():
    """Let the builders import without the SDK installed; none is called here."""
    if "openai" in sys.modules:
        return
    try:
        import openai  # noqa: F401
    except ImportError:
        stub = types.ModuleType("openai")
        stub.OpenAI = type("OpenAI", (), {"__init__": lambda self, *a, **k: None})
        sys.modules["openai"] = stub


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _builders():
    _stub_openai()
    gen = _load("_qimg7_gen", "code/generation/run_openai_baseline.py")
    selfcheck = _load("_qimg7_selfcheck", "code/routing/run_selfcheck_gate.py")
    triage = _load("_qimg7_triage", "code/routing/run_triage_gate.py")
    resolver = _load("_qimg7_sar", "code/routing/run_source_aware_resolver.py")
    judge = _load("_qimg7_judge", "code/eval/judge_against_clean.py")
    return {
        "answer": lambda row, ctx: gen.build_text_prompt(row),
        "answer_content": lambda row, ctx: json.dumps(
            gen.build_request_content(row, include_pixels=False)[0]),
        "selfcheck_gate": lambda row, ctx: selfcheck.make_gate_prompt(row),
        "triage_gate": lambda row, ctx: triage.make_prompt(row),
        "source_aware": lambda row, ctx: resolver.make_prompt(
            row, ctx["parametric"], ctx["text_only"], ctx["full_mm"], True, None),
        "judge": lambda row, ctx: (
            judge.build_reference_text(ctx["clean_row"]) if ctx["clean_row"] else ""),
    }, judge


def _rows_and_context(judge):
    for dataset, n in DATASETS:
        bench = os.path.join(
            REPO, f"benchmark/evaluated/{dataset}_qimg7_mm_regimes_{n}.jsonl")
        if not os.path.exists(bench):
            raise SystemExit(f"missing benchmark: {bench}")
        rows = [json.loads(line) for line in open(bench, encoding="utf-8")]
        answers = {}
        for baseline in ("parametric", "text_only", "full_mm"):
            path = os.path.join(
                REPO, f"evaluation/predictions/gpt4omini/{dataset}/{baseline}_{n}.jsonl")
            answers[baseline] = {
                r["row_id"]: (r.get("answer") or "")
                for r in map(json.loads, open(path, encoding="utf-8"))
            } if os.path.exists(path) else {}
        clean = judge.build_clean_reference_map(bench)
        for row in rows:
            rid = row["row_id"]
            yield dataset, row, {
                "parametric": answers["parametric"].get(rid, ""),
                "text_only": answers["text_only"].get(rid, ""),
                "full_mm": answers["full_mm"].get(rid, ""),
                "clean_row": clean.get(int(row["qid"])),
            }


def test_detector_is_not_vacuous():
    """A raw row must trip the detector, or a clean result proves nothing."""
    bench = os.path.join(REPO, "benchmark/evaluated/fava_qimg7_mm_regimes_20.jsonl")
    row = json.loads(open(bench, encoding="utf-8").readline())
    findings = find_leakage(json.dumps(row), row)
    assert findings, "detector found nothing in a raw benchmark row; it is broken"


def test_unknown_component_is_rejected():
    """A typo must raise, not silently fall back to a permissive allow-list."""
    row = {"question": "q", "text_evidence": [], "image_evidence": {}}
    try:
        model_facing_evidence(row, "not_a_component")
    except KeyError:
        return
    raise AssertionError("unknown component did not raise")


def test_no_construction_data_in_any_prompt():
    builders, judge = _builders()
    failures, checked = [], 0
    for dataset, row, ctx in _rows_and_context(judge):
        for name, build in builders.items():
            text = build(row, ctx)
            if not text:
                continue
            checked += 1
            for finding in find_leakage(text, row):
                failures.append(f"{dataset} row_id={row['row_id']} [{name}] {finding}")
    assert checked, "no prompts were checked"
    assert not failures, (
        f"{len(failures)} leak(s) across {checked} prompts:\n  "
        + "\n  ".join(failures[:20])
    )
    return checked


def main():
    test_detector_is_not_vacuous()
    print("ok  detector is not vacuous")
    test_unknown_component_is_rejected()
    print("ok  unknown component rejected")
    checked = test_no_construction_data_in_any_prompt()
    print(f"ok  no construction data in any of {checked} prompts "
          f"({len(ALLOW_LISTS)} components x {sum(n for _, n in DATASETS) * 16} rows)")
    print("PASS")


if __name__ == "__main__":
    main()
