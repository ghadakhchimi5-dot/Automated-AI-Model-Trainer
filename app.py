from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from auto_pipeline import AutoRunResult, run_automatic_training
from core.data_utils import save_uploaded_files
from core.hardware import detect_hardware
from core.inference import classify_image, generate_text, load_model_package
from core.packaging import unpack_model_zip

st.set_page_config(page_title="Automated Fine-Tuning Platform", layout="wide")

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
LOADED_DIR = OUTPUT_DIR / "loaded_packages"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOADED_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_CSS = """
<style>
.ghd-hero {
    background: linear-gradient(135deg, #6C5CE7 0%, #8E7CF1 45%, #A29BFE 100%);
    padding: 2.1rem 2.4rem;
    border-radius: 18px;
    color: #fff;
    margin-bottom: 1.6rem;
    box-shadow: 0 12px 30px -12px rgba(108, 92, 231, 0.55);
}
.ghd-hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    padding: 0.15rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 0.6rem;
}
.ghd-hero h1 { margin: 0 0 0.35rem 0; font-size: 2.05rem; line-height: 1.15; }
.ghd-hero p { margin: 0; opacity: 0.92; font-size: 1.02rem; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    box-shadow: 0 4px 18px -8px rgba(30, 27, 46, 0.12);
}

[data-testid="stSidebar"] { border-right: 1px solid rgba(108, 92, 231, 0.12); }

[data-testid="stMetric"] {
    background: #F5F3FF;
    border-radius: 12px;
    padding: 0.7rem 0.9rem 0.5rem 0.9rem;
    border: 1px solid rgba(108, 92, 231, 0.12);
}
[data-testid="stMetricLabel"] { opacity: 0.75; }

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C5CE7, #8E7CF1);
    border: none;
    box-shadow: 0 8px 18px -8px rgba(108, 92, 231, 0.6);
    font-weight: 600;
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.06); }

[data-testid="stTabs"] button[role="tab"] { font-weight: 600; }

.ghd-badge {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    vertical-align: middle;
}
.badge-pass { background: #E4F8EF; color: #0E9F6E; border: 1px solid rgba(14,159,110,0.2); }
.badge-fail { background: #FDECEC; color: #E02424; border: 1px solid rgba(224,36,36,0.2); }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <div class="ghd-hero">
            <div class="ghd-hero-badge">AUTOML STUDIO</div>
            <h1>Automated Fine-Tuning Platform</h1>
            <p>One-click automation for text QA/chat SLM fine-tuning and image classification.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Hardware")
        hw = detect_hardware()
        with st.container(border=True):
            st.metric("Tier", hw.tier)
            st.metric("GPU", hw.gpu_name or "CPU only")
            c1, c2 = st.columns(2)
            c1.metric("VRAM", f"{hw.vram_gb:.1f} GB")
            c2.metric("RAM", f"{hw.ram_gb:.1f} GB")

        st.markdown("### About")
        with st.container(border=True):
            st.markdown(
                "**Automated Fine-Tuning Platform** turns a raw dataset into a "
                "ready-to-use model with a single click."
            )
            st.markdown(
                "- **Auto model & strategy** — picks the base model and PEFT method for your task\n"
                "- **Auto hyperparameters** — batch size and training budget sized to your hardware\n"
                "- **Built-in evaluation** — metrics and pass/fail verdict generated after every run\n"
                "- **Portable output** — download a self-contained model package, or load one back in"
            )
            st.caption("No multi-step configuration required.")


METRIC_SPECS: dict[str, dict[str, str]] = {
    "accuracy": {"label": "Accuracy", "group": "Quality", "kind": "percent", "help": "Share of correctly predicted samples."},
    "top_5_accuracy": {"label": "Top-5 Accuracy", "group": "Quality", "kind": "percent", "help": "Correct label found within the top 5 predictions."},
    "precision": {"label": "Precision", "group": "Quality", "kind": "percent", "help": "Of predicted positives, share that were correct."},
    "recall": {"label": "Recall", "group": "Quality", "kind": "percent", "help": "Of actual positives, share that were found."},
    "f1": {"label": "F1 Score", "group": "Quality", "kind": "percent", "help": "Harmonic mean of precision and recall."},
    "roc_auc": {"label": "ROC AUC", "group": "Quality", "kind": "percent", "help": "Area under the ROC curve."},
    "roc_auc_micro": {"label": "ROC AUC (micro)", "group": "Quality", "kind": "percent", "help": "Micro-averaged ROC AUC across classes."},
    "pr_auc": {"label": "PR AUC", "group": "Quality", "kind": "percent", "help": "Area under the precision-recall curve."},
    "token_accuracy": {"label": "Token Accuracy", "group": "Quality", "kind": "percent", "help": "Share of correctly predicted tokens."},
    "eval_loss": {"label": "Eval Loss", "group": "Loss & Perplexity", "kind": "float", "help": "Cross-entropy loss on the evaluation set (lower is better)."},
    "perplexity": {"label": "Perplexity", "group": "Loss & Perplexity", "kind": "float", "help": "Exponential of eval loss (lower is better)."},
    "bleu": {"label": "BLEU", "group": "Text Generation", "kind": "raw", "help": "N-gram overlap with reference answers, 0-100 (higher is better)."},
    "rouge1": {"label": "ROUGE-1", "group": "Text Generation", "kind": "raw", "help": "Unigram overlap with reference answers."},
    "rouge2": {"label": "ROUGE-2", "group": "Text Generation", "kind": "raw", "help": "Bigram overlap with reference answers."},
    "rougeL": {"label": "ROUGE-L", "group": "Text Generation", "kind": "raw", "help": "Longest common subsequence overlap with reference answers."},
    "rougeLsum": {"label": "ROUGE-Lsum", "group": "Text Generation", "kind": "raw", "help": "ROUGE-L computed over summary sentences."},
    "token_count": {"label": "Tokens Evaluated", "group": "Other", "kind": "int", "help": "Number of tokens used to compute token accuracy."},
    "generation_sample_size": {"label": "Samples Evaluated", "group": "Other", "kind": "int", "help": "Number of generated answers scored."},
    "task_type": {"label": "Task Type", "group": "Other", "kind": "raw"},
}
METRIC_GROUP_ORDER = ["Quality", "Loss & Perplexity", "Text Generation", "Other"]


def format_metric_value(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "percent" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value * 100:.2f}%"
    if kind == "float" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4f}"
    if kind == "int" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{int(value)}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_metrics(scalar_metrics: dict[str, Any]) -> None:
    grouped: dict[str, list[tuple[str, Any]]] = {group: [] for group in METRIC_GROUP_ORDER}
    for key, value in scalar_metrics.items():
        spec = METRIC_SPECS.get(key, {"label": key.replace("_", " ").title(), "group": "Other", "kind": "raw"})
        grouped[spec.get("group", "Other")].append((key, value))

    st.markdown("**Metrics**")
    for group in METRIC_GROUP_ORDER:
        entries = grouped.get(group, [])
        if not entries:
            continue
        st.caption(group)
        cols = st.columns(min(4, len(entries)))
        for i, (key, value) in enumerate(entries):
            spec = METRIC_SPECS.get(key, {"label": key.replace("_", " ").title(), "kind": "raw"})
            cols[i % len(cols)].metric(
                spec["label"],
                format_metric_value(value, spec.get("kind", "raw")),
                help=spec.get("help"),
            )


def verdict_badge(verdict: str) -> str:
    label = (verdict or "unknown").strip()
    cls = "badge-pass" if label.lower() == "pass" else "badge-fail"
    return f'<span class="ghd-badge {cls}">{label.upper()}</span>'


def render_run_results(result: AutoRunResult) -> None:
    st.markdown(
        f"#### Run summary &nbsp; {verdict_badge(result.evaluation_verdict)}",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Task", result.task)
        c2.metric("Evaluation", result.evaluation_verdict.upper())
        c3.metric("Run folder", Path(result.run_dir).name)

        scalar_metrics = {
            k: v
            for k, v in result.metrics.items()
            if isinstance(v, (int, float, str)) and not isinstance(v, bool) and k != "roc_curve_plot"
        }
        if scalar_metrics:
            render_metrics(scalar_metrics)

        roc_curve_plot = result.metrics.get("roc_curve_plot")
        if roc_curve_plot:
            roc_curve_path = Path(result.run_dir) / roc_curve_plot
            if roc_curve_path.exists():
                st.markdown("**ROC Curve**")
                st.image(str(roc_curve_path), use_container_width=True)

    with st.expander("Full run details (manifest, plan, report)", expanded=False):
        st.json(result.to_dict())

    zip_bytes = Path(result.zip_path).read_bytes()
    st.download_button(
        "Download final model package",
        data=zip_bytes,
        file_name=Path(result.zip_path).name,
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )


@st.cache_resource(show_spinner=False)
def cached_load_package(package_dir: str):
    return load_model_package(package_dir)


def render_package_inference(package_dir: str) -> None:
    try:
        package = cached_load_package(package_dir)
    except Exception as exc:
        st.error(f"Could not load package: {exc}")
        return

    st.success(f"Loaded package: `{package.task}`")
    manifest = package.manifest
    with st.expander("Package manifest", expanded=False):
        st.json(manifest)

    if package.task == "text_qa":
        st.subheader("Chat with the fine-tuned SLM")
        with st.container(border=True):
            # Reset the conversation when a different package is loaded, so stale
            # turns from another model don't leak into the context window.
            if st.session_state.get("chat_history_pkg") != package_dir:
                st.session_state["chat_history"] = []
                st.session_state["chat_history_pkg"] = package_dir
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            if st.session_state["chat_history"] and st.button("Clear conversation"):
                st.session_state["chat_history"] = []
                st.rerun()
            prompt = st.chat_input("Ask something related to the fine-tuned data...")
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            if prompt:
                history = list(st.session_state["chat_history"])
                st.session_state["chat_history"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Generating..."):
                        result = generate_text(package, prompt, history=history)
                    st.markdown(result["answer"])
                st.session_state["chat_history"].append({"role": "assistant", "content": result["answer"]})

    elif package.task == "image_classification":
        st.subheader("Classify an image")
        with st.container(border=True):
            image_file = st.file_uploader(
                "Upload an image for inference",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"infer_image_{package_dir}",
            )
            if image_file is not None:
                image = Image.open(image_file).convert("RGB")
                col_img, col_preds = st.columns([1, 1.4])
                col_img.image(image, caption="Input image", width=320)
                with st.spinner("Classifying..."):
                    preds = classify_image(package, image, top_k=5)
                preds_pct = [{"label": p["label"], "confidence": p["score"] * 100} for p in preds]
                with col_preds:
                    st.dataframe(
                        preds_pct,
                        column_config={
                            "label": st.column_config.TextColumn("Class"),
                            "confidence": st.column_config.ProgressColumn(
                                "Confidence", min_value=0.0, max_value=100.0, format="%.1f%%"
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )


def render_train_tab() -> None:
    with st.container(border=True):
        st.subheader("1. Dataset & goal")
        uploaded = st.file_uploader(
            "Upload user dataset",
            type=["csv", "json", "jsonl", "txt", "md", "zip"],
            help=(
                "Text QA: CSV/JSONL/TXT/MD. Image classification: zip with class folders. "
                "Upload multiple files to combine them (e.g. one zip per class, or several text files)."
            ),
            accept_multiple_files=True,
        )
        goal = st.text_area(
            "User goal",
            value="Create a specialized assistant from my data" if not uploaded else "",
            placeholder="Example: Create a support chatbot, or classify product defect images.",
        )

    with st.container(border=True):
        st.subheader("2. Configuration")
        c1, c2, c3 = st.columns(3)
        with c1:
            task = st.selectbox("Task", ["auto", "text_qa", "image_classification"], index=0)
        with c2:
            priority = st.selectbox("Priority", ["balanced", "performance", "memory"], index=0)
        with c3:
            st.caption("No multi-step configuration required. The pipeline selects model, PEFT, batch size and training budget automatically.")

        with st.expander("Optional overrides", expanded=False):
            override_model_id = st.text_input("Override base model ID", value="")
            max_samples = st.number_input("Max samples/images (0 = auto)", min_value=0, value=0, step=50)
            epochs = st.number_input("Epochs (0 = auto)", min_value=0, value=0, step=1)
            batch_size = st.number_input("Batch size (0 = auto)", min_value=0, value=0, step=1)
            max_length = st.number_input("Text max length (0 = auto)", min_value=0, value=0, step=128)
            merge_text_model = st.checkbox(
                "Also save merged text model when possible",
                value=False,
                help="The adapter package is much smaller. Merging can create a very large zip.",
            )

    if st.button("Run automatic training", type="primary", use_container_width=True, disabled=not uploaded):
        if not uploaded:
            st.error("Upload a dataset first.")
            return
        saved_path = save_uploaded_files(uploaded, UPLOAD_DIR)

        result: AutoRunResult | None = None
        error: Exception | None = None
        with st.status("Training pipeline is running. This may take time depending on model and hardware.", expanded=True) as status:

            def ui_log(msg: str) -> None:
                status.write(msg)

            try:
                result = run_automatic_training(
                    data_path=saved_path,
                    goal=goal,
                    task=task,
                    output_dir=OUTPUT_DIR,
                    priority=priority,
                    override_model_id=override_model_id.strip() or None,
                    epochs=int(epochs) if int(epochs) > 0 else None,
                    batch_size=int(batch_size) if int(batch_size) > 0 else None,
                    max_samples=int(max_samples) if int(max_samples) > 0 else None,
                    max_length=int(max_length) if int(max_length) > 0 else None,
                    merge_text_model=merge_text_model,
                    log_fn=ui_log,
                )
            except Exception as exc:
                error = exc
                status.update(label="Training failed", state="error", expanded=True)
            else:
                status.update(label="Training completed", state="complete", expanded=False)

        if error is not None:
            st.error(f"Training failed: {error}")
        elif result is not None:
            st.session_state["last_package_dir"] = str(result.run_dir)
            st.success("Training completed. The model package is ready.")
            render_run_results(result)

    last_dir = st.session_state.get("last_package_dir")
    if last_dir:
        st.divider()
        render_package_inference(last_dir)


def render_load_tab() -> None:
    with st.container(border=True):
        st.subheader("Load a generated model package")
        package_zip = st.file_uploader("Upload a previously downloaded model zip", type=["zip"], key="package_zip")
        if package_zip is not None and st.button("Load package", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_zip = Path(tmp) / package_zip.name
                tmp_zip.write_bytes(package_zip.getvalue())
                target = LOADED_DIR / Path(package_zip.name).stem
                if target.exists():
                    import shutil

                    shutil.rmtree(target)
                package_dir = unpack_model_zip(tmp_zip, target)
                st.session_state["loaded_package_dir"] = str(package_dir)
                cached_load_package.clear()
            st.success("Package loaded.")

    loaded_dir = st.session_state.get("loaded_package_dir")
    if loaded_dir:
        render_package_inference(loaded_dir)


def main() -> None:
    inject_css()
    render_hero()
    render_sidebar()

    tab_train, tab_load = st.tabs(["Train automatically", "Use / load package"])
    with tab_train:
        render_train_tab()
    with tab_load:
        render_load_tab()


if __name__ == "__main__":
    main()
