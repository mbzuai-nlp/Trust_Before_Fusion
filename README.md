# QIMG-7: Query-Image Pollution Benchmark (7 Attack Types)

QIMG-7 is a benchmark dataset for studying **query-image pollution** attacks against
multimodal retrieval-augmented generation (RAG) systems. For each question, the
image evidence retrieved alongside the query is adversarially manipulated using
**seven distinct attack types (T1–T7)**, producing polluted image–caption records
that test the robustness of vision-language models and multimodal retrievers.

> Anonymized release for peer review. All author- and institution-identifying
> information has been removed.

---

## Attack types

| ID | Name | Modifies | Saved image? |
|----|------|----------|--------------|
| T1 | Caption Flip | Caption/alt text only (false caption, original image URL) | No |
| T2 | Entity Swap | Image URL replaced with an unrelated donor image | No |
| T3 | Semantic Entity Rewrite | Visual scene edited (background/objects) via image editing | Yes |
| T4 | FigStep Typography | False claim rendered as text onto the image | Yes |
| T5 | Adversarial Patch (PGD) | CLIP-targeted adversarial patch embedded | Yes |
| T6 | Image Blend | Confusing donor image alpha-composited on top | Yes |
| T7 | Neural Style Transfer | Donor texture/style transferred onto content | Yes |

T1 and T2 alter only metadata (caption or image URL) and therefore have **no saved
image file** — `polluted_image_path` is empty for those rows. T3–T7 produce a
modified image file under `images/`.

Full method details (models, hyperparameters, fallbacks) are in
[`docs/IMAGE_POLLUTION_METHODS.md`](docs/IMAGE_POLLUTION_METHODS.md).

---

## Source datasets

QIMG-7 pollutes the query images of four long-form factuality datasets. Each
question yields up to 14 polluted records (7 attacks × 2 image slots).

| Dataset | Questions | Polluted records | Saved images |
|---------|-----------|------------------|--------------|
| AlpacaFact | 231 | 3,388 | 2,333 |
| Biography  | 182 | 2,548 | 1,839 |
| FAVA       | 100 | 1,400 | 1,016 |
| LongFact   | 248 | 3,550 | 2,499 |
| **Total**  | **761** | **10,886** | **7,687** |

---

## Repository layout

```
QIMG-7/
├── README.md
├── LICENSE
├── .gitattributes                       # Git LFS tracking for image files
├── docs/
│   └── IMAGE_POLLUTION_METHODS.md       # Full attack methodology
├── images/
│   ├── AlpacaFact/                      # T{3..7}_q{idx}_i{0,1}.jpg
│   ├── Biography/
│   ├── FAVA/
│   └── LongFact/
├── AlpacaFact_query_image_polluted.csv
├── Biography_query_image_polluted.csv
├── FAVA_query_image_polluted.csv
└── LongFact_query_image_polluted.csv
```

Image filenames follow `T{type}_q{question_idx}_i{image_idx}.jpg`, e.g.
`images/FAVA/T3_q1_i0.jpg` is the Semantic Entity Rewrite of question 1, image 0.

All `polluted_image_path` values in the CSVs are **relative to the repository
root**, so the CSVs and `images/` must remain siblings.

---

## CSV schema

Each `*_query_image_polluted.csv` has these columns:

| Column | Description |
|--------|-------------|
| `question_idx` | Index of the source question within its dataset |
| `question` | The user question text |
| `image_idx` | Which retrieved image slot (0 or 1) was polluted |
| `pollution_type` | Attack applied (`caption_flip`, `entity_swap`, `semantic_entity_rewrite`, `figstep_typography`, `adversarial_patch`, `image_blend`, `neural_style_transfer`) |
| `original_image_url` | URL of the original retrieved image |
| `original_alt` | Original alt/caption text |
| `polluted_image_url` | Polluted image URL (T1: unchanged; T2: donor URL) |
| `polluted_alt` | Polluted caption/alt text |
| `polluted_image_path` | Relative path to the saved polluted image (empty for T1/T2) |
| `manipulation_method` | How the pollution was produced |
| `manipulation_prompt` | Prompt/instruction used (where applicable) |
| `donor_question_idx` | Source question for donor-based attacks (T2/T6) |
| `rationale` | Notes on the manipulation |

The FAVA question 0 (`question_idx == 0`) records are a hand-curated, qualitatively
refined version of the seven attacks (useful for figures); they follow the same
schema and filename convention as every other question and live in the standard
`FAVA_query_image_polluted.csv` / `images/FAVA/` location.

---

## Usage

```python
import csv, os
from PIL import Image

rows = list(csv.DictReader(open("FAVA_query_image_polluted.csv")))
row = next(r for r in rows if r["polluted_image_path"])   # first attack with a saved image
img = Image.open(row["polluted_image_path"])              # path is repo-root relative
print(row["pollution_type"], row["polluted_alt"])
img.show()
```

---

## Obtaining the images (Git LFS)

The image files (~1.9 GB) are tracked with **Git LFS**. To clone with images:

```bash
git lfs install
git clone <repo-url>
```

A standalone archive (`QIMG-7.zip`) containing the full tree is also distributed
for hosts that do not support Git LFS (e.g. Zenodo, where it is published with a
DOI). If you cloned without LFS support you will see small pointer files instead
of images — install Git LFS and run `git lfs pull`.

---

## License

CSV annotations and metadata are released under **CC BY 4.0** (see `LICENSE`).
The polluted images are derivative works of publicly available source images
referenced by `original_image_url`; downstream use should respect the licenses of
those original sources. This dataset is provided for **research on multimodal RAG
robustness and security**.
