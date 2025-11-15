# app.py
import io
import json
import logging
import os
import traceback
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
from torchvision import transforms
import pickle

# --- logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fashion_app")

# --- app + templates ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- config / paths ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_quantized_scripted.pt"
MLB_PATH = BASE_DIR / "model" / "mlb_dict.pkl"

# --- helper utils ---
def load_model(device: torch.device):
    logger.info(f"Loading scripted model from {MODEL_PATH}")
    model = torch.jit.load(str(MODEL_PATH), map_location=device)
    model.eval()
    logger.info("Loaded scripted model successfully.")
    return model

def load_mlb(pkl_path: Path):
    logger.info(f"Loading mlb (labels) from {pkl_path}")
    with open(pkl_path, "rb") as f:
        mlb = pickle.load(f)
    # Expecting mlb.classes_ or similar structure; try both
    if hasattr(mlb, "classes_"):
        classes = list(mlb.classes_)
    elif isinstance(mlb, dict) and "classes" in mlb:
        classes = list(mlb["classes"])
    else:
        # fallback: if mlb is list-like
        try:
            classes = list(mlb)
        except Exception:
            classes = []
    logger.info(f"Attributes loaded: {classes}")
    return classes

# --- device + model load at startup ---
DEVICE = torch.device("cpu")
try:
    model = load_model(DEVICE)
except Exception as e:
    logger.exception("Failed to load model at startup.")
    model = None

try:
    attribute_list = load_mlb(MLB_PATH)
except Exception as e:
    logger.exception("Failed to load mlb labels at startup.")
    attribute_list = []

# --- transforms (adjust if your training used different sizes) ---
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --- routes ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # serve index.html (your existing template can use "predictions")
    return templates.TemplateResponse("index.html", {"request": request, "predictions": None})

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Predict endpoint. Accepts multipart/form-data with field "file".
    This function wraps inference in try/except and returns error details on failure.
    """
    logger.info("Received /predict request")
    # save uploaded file to /tmp for debugging/inspection
    try:
        contents = await file.read()
        tmp_dir = Path("/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"uploaded_{file.filename.replace(' ', '_')}"
        with open(tmp_path, "wb") as f:
            f.write(contents)
        logger.info(f"Saved uploaded file to {tmp_path} ({len(contents)} bytes)")
    except Exception as e:
        logger.exception("Failed to read or save uploaded file")
        return JSONResponse({"error": "failed to read uploaded file", "detail": str(e)}, status_code=500)

    # perform inference in try/except to capture crashes
    try:
        # verify model present
        if model is None:
            raise RuntimeError("Model not loaded on server. Check startup logs.")

        # open image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        x = transform(image).unsqueeze(0)  # shape: (1, C, H, W)
        logger.info(f"Transformed image tensor shape: {x.shape}")

        # inference
        with torch.no_grad():
            # ensure on device
            x = x.to(DEVICE)
            logger.info("Running model forward pass")
            out = model(x)
            # handle common outputs: tensor or dict
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, torch.Tensor):
                probs = out.squeeze(0).cpu().numpy()
            elif isinstance(out, dict) and "probs" in out:
                probs = out["probs"].squeeze(0).cpu().numpy()
            else:
                # try to convert to numpy
                try:
                    probs = torch.tensor(out).squeeze(0).cpu().numpy()
                except Exception:
                    raise RuntimeError(f"Unexpected model output type: {type(out)}")

        # map to attribute labels
        if len(attribute_list) == 0:
            labels = [f"attr_{i}" for i in range(len(probs))]
        else:
            labels = attribute_list
            # if mismatch in length, fallback to numeric labels
            if len(labels) != len(probs):
                logger.warning("Length mismatch between labels and model output; falling back to numeric label names.")
                labels = [f"attr_{i}" for i in range(len(probs))]

        # build predictions list
        preds: List[Dict[str, Any]] = []
        for lbl, p in zip(labels, probs):
            preds.append({"label": str(lbl), "score": float(p)})

        logger.info("Inference complete; returning result to template")

        # Render same template but include predictions
        return templates.TemplateResponse("index.html", {"request": request, "predictions": preds})

    except Exception as exc:
        # log full traceback to server logs (Render)
        tb = traceback.format_exc()
        logger.error("Exception during prediction:\n" + tb)
        # return error information as JSON for debugging (safe to remove later)
        return JSONResponse(
            {"error": "prediction failed", "exception": str(exc), "traceback": tb},
            status_code=500
        )
