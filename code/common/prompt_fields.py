"""Single allow-list for what a QIMG-7 benchmark row may put in a model prompt.

Benchmark rows carry the construction record alongside the evidence: `regime`,
`text_status`, `image_status`, `image_pollution_type`, and the nested
`image_evidence.metadata` block with `manipulation_method`,
`manipulation_prompt`, `donor_question_idx`, `rationale`, `original_alt`, and
`polluted_alt`. None of it may reach an answer, router, or judge prompt --
a model that can read the attack label is not being tested on the attack.

Before this module each component re-derived that boundary independently. They
all agreed, but nothing enforced the agreement. This module holds the allow-list
and the extraction; each component keeps its own rendering, so wiring it in
changes no prompt text.

`local_path` and `resolved_local_path` are excluded deliberately and are worth
naming: a packaged path like `images/FAVA/T6_q18_i1.jpg` encodes the attack
family in its filename, so putting one in a prompt would leak the label even
though the field looks innocuous. Those fields are for loading pixels only.
"""

from copy import deepcopy

# Fields a component may read out of a benchmark row.
_DEFAULT_ALLOW = {
    "row": ("question",),
    "text_evidence": ("title", "snippet", "url"),
    "image_evidence": ("title", "alt_text", "page_url"),
}

# Per-component allow-lists. All current components share the default: the judge
# differs by being handed the clean TC_IC row for the qid, not by seeing extra
# fields. Kept per-component so a future divergence is a deliberate edit here
# rather than a quiet drift in one script.
ALLOW_LISTS = {
    "answer": _DEFAULT_ALLOW,          # code/generation/run_openai_baseline.py
    "selfcheck_gate": _DEFAULT_ALLOW,  # code/routing/run_selfcheck_gate.py
    "triage_gate": _DEFAULT_ALLOW,     # code/routing/run_triage_gate.py
    "source_aware": _DEFAULT_ALLOW,    # code/routing/run_source_aware_resolver.py
    "judge": _DEFAULT_ALLOW,           # code/eval/judge_against_clean.py
}

# Construction fields that must never appear in a prompt, by key name.
FORBIDDEN_KEYS = (
    "regime", "text_status", "image_status", "image_pollution_type",
    "pollution_type", "manipulation_method", "manipulation_prompt",
    "donor_question_idx", "rationale", "original_alt", "polluted_alt",
    "local_path", "resolved_local_path",
)

# Forbidden fields whose *values* are distinctive enough to search for directly.
# `original_alt` / `polluted_alt` are excluded: on clean and polluted rows
# respectively their value equals the allowed `alt_text`, so a value search would
# flag them permanently. `pollution_type` is excluded because "clean" is an
# ordinary word. Both are still covered by the key check above.
FORBIDDEN_VALUE_FIELDS = (
    "manipulation_method", "manipulation_prompt", "donor_question_idx",
    "rationale", "local_path", "resolved_local_path",
)
FORBIDDEN_TOP_LEVEL_VALUE_FIELDS = ("regime", "text_status", "image_status")

MIN_VALUE_LEN = 8


def _text_evidence(row, allow, limit):
    items = row.get("text_evidence") or []
    if limit is not None:
        items = items[:limit]
    return [{k: str(t.get(k, "") or "") for k in allow["text_evidence"]} for t in items]


def model_facing_evidence(row, component="answer", text_limit=None):
    """Return only the fields `component` is allowed to put in a prompt.

    Raises KeyError for an unknown component rather than silently defaulting,
    so a typo cannot widen the allow-list.
    """
    allow = ALLOW_LISTS[component]
    image = row.get("image_evidence") or {}
    return {
        **{k: str(row.get(k, "") or "") for k in allow["row"]},
        "text_evidence": _text_evidence(row, allow, text_limit),
        "image_evidence": {
            k: str(image.get(k, "") or "") for k in allow["image_evidence"]
        },
    }


def find_leakage(text, row):
    """Return a list of findings describing construction data present in `text`."""
    findings = []
    image = row.get("image_evidence") or {}
    metadata = image.get("metadata") or {}

    for key in FORBIDDEN_KEYS:
        if key in text:
            findings.append(f"forbidden key name {key!r} appears in the prompt")

    for field in FORBIDDEN_VALUE_FIELDS:
        value = str(metadata.get(field, "") or image.get(field, "") or "").strip()
        if len(value) >= MIN_VALUE_LEN and value in text:
            findings.append(
                f"value of image_evidence.{field} appears in the prompt: {value[:60]!r}"
            )

    for field in FORBIDDEN_TOP_LEVEL_VALUE_FIELDS:
        value = str(row.get(field, "") or "").strip()
        if len(value) >= MIN_VALUE_LEN and value in text:
            findings.append(f"value of {field} appears in the prompt: {value!r}")

    return findings


def assert_no_leakage(text, row, component=""):
    findings = find_leakage(text, row)
    if findings:
        where = f" ({component})" if component else ""
        raise AssertionError(
            f"prompt leaks benchmark construction data{where} for row_id="
            f"{row.get('row_id')}:\n  " + "\n  ".join(findings)
        )


def redacted_row(row, component="answer"):
    """A deep copy of `row` with every non-allow-listed field removed."""
    return deepcopy(model_facing_evidence(row, component))
