# QIMG-7 Release Candidate Bundle

This bundle is generated from internal MM-MIRAGE artifacts for public QIMG-7 release hardening.

## Included Content
- `benchmark/pool`: candidate pool metadata CSVs (four datasets).
- `benchmark/evaluated`: evaluated benchmark JSONL files with sanitized image paths.
- `evaluation/*/gpt4omini`: generated answers, routing choices, judged outputs, and summary tables.
- `metadata`: filtering flow counts and implementation detail addendum.
- `evaluation/human_validation`: completed 96-item human annotation packet.
- `code`: cleaned release scripts with comments removed from shell/python files.

## Notes
- API keys are environment-variable based only.
- Absolute local paths are removed or converted to relative `images/...` paths.
- This folder is intended to be copied into the public `oa07610/QIMG-7` repository.

## File Count
- Total copied files: 141
