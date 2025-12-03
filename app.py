from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_MODEL = MODEL_DIR / "yolov8n.pt"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
PRED_DIR = STATIC_DIR / "predictions"
ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]


def _ensure_directories() -> None:
    for folder in (UPLOAD_DIR, PRED_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

_ensure_directories()
model_cache: Dict[str, YOLO] = {}


def _load_model(path: Path) -> YOLO:
    """Load and cache a YOLO model by path."""
    key = str(path.resolve())
    if key not in model_cache:
        model_cache[key] = YOLO(key)
    return model_cache[key]


def list_models() -> List[Path]:
    """List available model files in the model directory."""
    if not MODEL_DIR.exists():
        return []
    return sorted(p for p in MODEL_DIR.iterdir() if p.suffix in {".pt", ".onnx", ".engine"})


def run_inference(model: YOLO, image_path: Path, output_path: Path) -> Dict[str, int]:
    """Run the YOLO model on the given image and save annotated output."""
    results = model(str(image_path))
    result = results[0]
    result.save(filename=str(output_path))

    counts: Dict[str, int] = {}
    for cls_idx in result.boxes.cls.tolist():
        class_id = int(cls_idx)
        class_name = model.names.get(class_id, f"class_{class_id}")
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


@app.route("/", methods=["GET", "POST"])
def index():
    available_models = list_models()
    context = {
        "original_url": None,
        "prediction_url": None,
        "counts": None,
        "error": None,
        "available_models": available_models,
        "selected_model": None,
    }

    if request.method == "POST":
        upload = request.files.get("image")
        selected_model_name = request.form.get("model_path")
        selected_model_path = None

        if not upload or upload.filename == "":
            context["error"] = "Please choose an image to upload."
            return render_template("index.html", **context)

        if not _is_allowed(upload.filename):
            context["error"] = f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            return render_template("index.html", **context)

        if selected_model_name:
            selected_model_path = MODEL_DIR / selected_model_name
        elif available_models:
            selected_model_path = DEFAULT_MODEL if DEFAULT_MODEL.exists() else available_models[0]

        if not selected_model_path or not selected_model_path.exists():
            context["error"] = "Selected model not found. Please choose a valid model file."
            return render_template("index.html", **context)

        unique_name = f"{uuid.uuid4().hex}{Path(upload.filename).suffix.lower()}"
        saved_path = UPLOAD_DIR / unique_name
        pred_path = PRED_DIR / unique_name

        upload.save(saved_path)

        try:
            model = _load_model(selected_model_path)
            counts = run_inference(model, saved_path, pred_path)
        except Exception as exc:  # pragma: no cover
            context["error"] = f"Inference failed: {exc}"
            return render_template("index.html", **context)

        context.update(
            original_url=url_for("static", filename=f"uploads/{unique_name}"),
            prediction_url=url_for("static", filename=f"predictions/{unique_name}"),
            counts=counts,
            selected_model=selected_model_path.name,
        )

    return render_template("index.html", **context)


def _process_upload(file_storage, selected_model_name: str | None) -> Tuple[str, str, Dict[str, int]]:
    available_models = list_models()
    if selected_model_name:
        selected_model_path = MODEL_DIR / selected_model_name
    elif available_models:
        selected_model_path = DEFAULT_MODEL if DEFAULT_MODEL.exists() else available_models[0]
    else:
        raise ValueError("No available models.")

    if not selected_model_path.exists():
        raise FileNotFoundError("Selected model not found.")

    unique_name = f"{uuid.uuid4().hex}{Path(file_storage.filename).suffix.lower()}"
    saved_path = UPLOAD_DIR / unique_name
    pred_path = PRED_DIR / unique_name
    file_storage.save(saved_path)

    model = _load_model(selected_model_path)
    counts = run_inference(model, saved_path, pred_path)

    return (
        url_for("static", filename=f"uploads/{unique_name}"),
        url_for("static", filename=f"predictions/{unique_name}"),
        counts,
    )


@app.route("/predict", methods=["POST"])
def predict():
    upload = request.files.get("image")
    selected_model_name = request.form.get("model_path")

    if not upload or upload.filename == "":
        return jsonify({"error": "请上传图片"}), 400
    if not _is_allowed(upload.filename):
        return jsonify({"error": f"仅支持: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    try:
        original_url, prediction_url, counts = _process_upload(upload, selected_model_name)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "original_url": original_url,
            "prediction_url": prediction_url,
            "counts": counts,
        }
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):  # pragma: no cover
    return render_template(
        "index.html",
        original_url=None,
        prediction_url=None,
        counts=None,
        error="上传的图片太大，请压缩后再试（限制 20MB）。",
        available_models=list_models(),
        selected_model=None,
    ), 413


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Delete cached upload/prediction files except those explicitly kept."""
    payload = request.get_json(force=True, silent=True) or {}
    keep_files = {Path(name).name for name in payload.get("keep", []) if name}
    removed = []
    for folder in (UPLOAD_DIR, PRED_DIR):
        if not folder.exists():
            continue
        for file in folder.iterdir():
            if not file.is_file():
                continue
            if file.name in keep_files:
                continue
            try:
                file.unlink()
                removed.append(file.name)
            except OSError:
                continue
    return jsonify({"removed": removed, "kept": list(keep_files)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
