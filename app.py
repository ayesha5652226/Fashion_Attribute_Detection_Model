# app.py
import io
import base64
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
import torch
import torchvision.transforms as T
import pickle

LOG = logging.getLogger("fashion_app")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_quantized_scripted.pt"
MLB_PATH = BASE_DIR / "model" / "mlb_dict.pkl"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Try to load model and label binarizer (MLB). Fail gracefully if missing.
device = torch.device("cpu")
model = None
mlb = None
attributes_list: Optional[List[str]] = None

def load_model_and_labels():
    global model, mlb, attributes_list
    try:
        if MODEL_PATH.exists():
            LOG.info(f"Loading scripted model from {MODEL_PATH}")
            model = torch.jit.load(str(MODEL_PATH), map_location=device)
            model.eval()
            LOG.info("Loaded scripted model successfully.")
        else:
            LOG.warning(f"Model file not found at {MODEL_PATH} — predictions will fail until you add it.")
        if MLB_PATH.exists():
            LOG.info(f"Loading mlb (labels) from {MLB_PATH}")
            with open(MLB_PATH, "rb") as f:
                mlb = pickle.load(f)
            # If mlb has classes_, use that. Otherwise allow a saved attributes list.
            if hasattr(mlb, "classes_"):
                attributes_list = list(mlb.classes_)
            else:
                # optional fallback: try to read a plain txt list in the model folder
                attributes_list = None
            LOG.info("Loaded mlb/labels.")
        else:
            LOG.warning(f"MLB file not found at {MLB_PATH} — labels will be numeric indices.")
    except Exception as e:
        LOG.exception("Error loading model/labels: %s", e)

load_model_and_labels()

# image transforms (matches typical training pipeline)
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render home page with upload form.
    """
    return templates.TemplateResponse("index.html", {"request": request, "preds": None})

@app.get("/health")
async def health():
    return {"status": "ok"}

def pil_to_base64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Accept an uploaded image, run model inference, and return rendered template.
    """
    LOG.info("Received /predict request")
    # Save uploaded file to a temporary path
    try:
        content = await file.read()
        temp_path = Path("/tmp") / f"uploaded_{Path(file.filename).name}"
        with open(temp_path, "wb") as f:
            f.write(content)
        LOG.info("Saved uploaded file to %s (%d bytes)", temp_path, temp_path.stat().st_size)
    except Exception:
        LOG.exception("Failed to save uploaded file")
        return templates.TemplateResponse("index.html", {"request": request, "error": "Failed to read uploaded file."})

    # Load image and turn into model tensor
    try:
        pil = Image.open(temp_path).convert("RGB")
        tensor = transform(pil).unsqueeze(0)  # shape [1,3,224,224]
        LOG.info("Transformed image tensor shape: %s", tuple(tensor.shape))
    except Exception:
        LOG.exception("Failed to process image")
        return templates.TemplateResponse("index.html", {"request": request, "error": "Failed to process image."})

    # Run inference
    if model is None:
        LOG.error("Model not loaded")
        return templates.TemplateResponse("index.html", {"request": request, "error": "Model not loaded on server."})

    try:
        with torch.no_grad():
            LOG.info("Running model forward pass")
            out = model(tensor.to(device))
            # If model returns a tuple, take first element
            if isinstance(out, (list, tuple)):
                out = out[0]
            # Convert to probabilities. If logits, apply sigmoid; if already probs, clamp.
            try:
                probs = torch.sigmoid(out).cpu().squeeze().numpy()
            except Exception:
                probs = out.cpu().squeeze().numpy()
            LOG.info("Model forward pass done. Output length: %s", getattr(probs, "shape", "unknown"))
    except Exception:
        LOG.exception("Model inference failed")
        return templates.TemplateResponse("index.html", {"request": request, "error": "Model inference failed."})

    # Map labels
    results = []
    if attributes_list and len(attributes_list) == probs.shape[0]:
        for i, p in enumerate(probs):
            results.append({"label": attributes_list[i], "prob": float(p)})
    else:
        # length mismatch -> fallback numeric labels
        LOG.warning("Length mismatch between labels and model output; falling back to numeric label names.")
        for i, p in enumerate(probs):
            results.append({"label": str(i), "prob": float(p)})

    # Sort descending by prob
    results = sorted(results, key=lambda x: x["prob"], reverse=True)

    # Render image inline (base64) so it's visible on the returned page
    image_data = pil_to_base64(pil)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "preds": results,
        "image_data": image_data,
        "topk": results[:10],
    })
