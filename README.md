# GHD Automatic Trainer

This is a one-click replacement for the previous multi-page Streamlit demo. It automates:

1. dataset loading
2. task detection
3. cleaning and conversion
4. local hardware detection
5. base-model and PEFT strategy selection
6. training
7. evaluation
8. final model packaging
9. inference in the same interface

Supported use cases for this version:

- `text_qa`: turns user data into instruction/question-answer records and fine-tunes an SLM with LoRA/QLoRA.
- `image_classification`: fine-tunes an image classifier from a folder/zip of class folders.

## Install

```bash
cd ghd_auto_trainer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For NVIDIA GPU, install the PyTorch wheel matching your CUDA version before or after the requirements install.

## Streamlit app

```bash
streamlit run app.py
```

Upload a dataset, choose or keep `auto` task detection, write the user's goal, then click **Run automatic training**. When training finishes, download the generated model zip and use the inference panel.

## CLI

Text QA / chat SLM:

```bash
python auto_pipeline.py --task text_qa --data ./data/my_docs.csv --goal "Create a support chatbot" --output ./output/support_bot
```

Image classification:

```bash
python auto_pipeline.py --task image_classification --data ./data/images.zip --goal "Classify product defect images" --output ./output/defect_classifier
```

## Text dataset formats

Accepted: `.csv`, `.json`, `.jsonl`, `.txt`, `.md`, or a directory of text files.

Preferred columns are auto-detected:

- prompt side: `instruction`, `question`, `query`, `prompt`, `input`, `text`, `context`
- response side: `output`, `answer`, `response`, `target`, `completion`, `label`
- chat format: `messages`

If the data is plain text with no answer column, the pipeline creates simple synthetic QA pairs locally by chunking paragraphs and generating extractive questions.

## Image dataset formats

Preferred zip/folder structure:

```text
images/
  class_a/
    img1.jpg
    img2.jpg
  class_b/
    img3.jpg
```

A CSV with columns `image` and `label` is also supported when image paths are local.

## Output package

Each run creates:

- `manifest.json`
- `training_report.json`
- model files (`adapter_model.safetensors` for text PEFT or full classifier for images)
- tokenizer / processor files
- `<run_name>_model.zip`

The zip can be loaded back into the app for inference.
