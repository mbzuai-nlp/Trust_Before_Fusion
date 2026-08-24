# Human Validation

This directory contains the two QIMG-7 human-validation studies.

- `qimg7_human_validation_packet_96_completed.csv` contains human support labels for 96 sampled model outputs, using the same four-label rubric as the automatic judge.
- `image_attack_iaa/annotator_1.csv` and `annotator_2.csv` contain independent author annotations for 56 image attacks, balanced across four datasets and seven attack families. The three fields are `on_topic`, `fact_flipped`, and `plausible`, with values in `{yes, no, unclear}`.

Derived CSV summaries and `image_attack_iaa/iaa_results.txt` contain the released aggregate results. Regenerate and validate them with:

```bash
python code/eval/summarize_human_validation.py
python code/eval/compute_image_attack_iaa.py \
  evaluation/human_validation/image_attack_iaa/annotator_1.csv \
  evaluation/human_validation/image_attack_iaa/annotator_2.csv \
  evaluation/human_validation/image_attack_iaa/iaa_results.txt
```

To recreate the annotation interface for the fixed 56-item sample:

```bash
python code/eval/generate_image_attack_annotation_dashboard.py --output /tmp/qimg7_annotation_dashboard.html
```
