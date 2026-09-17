# How to apply this to the current project

The uploaded project currently uses six manual pages:

- `pages/1__Data.py`
- `pages/2__Cleaning.py`
- `pages/2__Configure.py`
- `pages/3__Train.py`
- `pages/4__Evaluate.py`
- `pages/5__Inference.py`

This package replaces that manual flow with one automatic flow:

- `app.py`: one-click Streamlit UI for upload -> train -> package -> inference.
- `auto_pipeline.py`: CLI and callable orchestration function.
- `core/data_utils.py`: task detection, data loading, text cleaning, synthetic QA generation, image-folder preparation.
- `core/hardware.py`: local CPU/GPU/VRAM detection and automatic training budget.
- `core/model_selector.py`: model + PEFT selection for text and image model selection.
- `core/text_trainer.py`: LoRA/QLoRA SLM training and packaging.
- `core/image_trainer.py`: image classifier training and packaging.
- `core/inference.py`: loading the generated zip/package for chat or image classification.
- `core/packaging.py`: manifest/report/zip helpers.

Recommended integration:

1. Copy the new `core/*.py`, `auto_pipeline.py`, and `requirements.txt` into your project.
2. Replace your current `app.py` with the new `app.py`, or add it as `Auto_App.py` if you want to keep the old step-by-step demo.
3. Keep your old pages only if you still want a manual debug mode.
4. Run `streamlit run app.py`.

The final downloaded zip is intentionally manifest-based. For text SLMs it usually contains the PEFT adapter rather than a full base model copy to avoid multi-GB downloads. The app can load it because the manifest stores the base model ID plus adapter path. If you need a standalone merged model, enable `merge_text_model=True` in the app advanced options or CLI `--merge-text-model`.
