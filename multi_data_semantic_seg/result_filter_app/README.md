# Result Filter App

Web UI to compare prediction masks vs ground truth and record pass/fail judgments. Updates the results JSON in real time with a `user_judgment` key on each annotation.

## Setup

```bash
cd multi_data_semantic_seg/result_filter_app
pip install -r requirements.txt
```

## Run

```bash
# From result_filter_app/
python app.py /path/to/results.json [--analysis /path/to/analysis.json] [--port 5000]

# Or use run.sh (paths relative to result_filter_app/)
bash run.sh ../output/run/results.json
bash run.sh ../output0/run_cityscapes/results_cityscapes.json ../output0/run_cityscapes/analysis.json
```

Then open **http://127.0.0.1:5000** in a browser.

## Usage

1. **Image** — Select an image (list is ordered by analysis variance if you passed `--analysis`).
2. **Instance type** — Choose `nCnI` or `1CnI`.
3. **Preview grid** — All predictions for that image are shown, ranked by `final_score` (best first). Each card shows a thumbnail and score.
4. **Compare each** — Click **Compare each (pop)** or any card to open the compare modal. Prediction mask and GT mask are shown side by side.
5. **Pass / No** — Click **Pass** or **No**; the backend adds `"user_judgment": "pass"` or `"user_judgment": "fail"` to that annotation in the results JSON and saves the file. The modal advances to the next prediction.
6. Judged annotations show a **pass** or **fail** badge on the preview card.

## API

- `GET /api/images` — List image ids.
- `GET /api/annotations?image_id=...&instance_type=nCnI` — Predictions + GT for that image (ranked by final_score).
- `POST /api/judge` — Body: `{"annotation_id": "seg_0", "user_judgment": "pass"|"fail"}`. Updates the annotation in the JSON on disk.
- `GET /files/masks/<filename>` — Serve mask images from `output_dir/masks/`.

Masks are resolved relative to the **directory containing the results JSON** (e.g. `results.json` and `masks/` must live in the same output folder).
