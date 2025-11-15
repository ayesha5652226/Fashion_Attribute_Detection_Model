# app.py
import io
import os
import logging
import pickle
import base64
from typing import List, Tuple, Optional

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fashion_app")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_PATH = os.path.join("model", "model_quantized_scripted.pt")  # adjust if different
MLB_PATH = os.path.join("model", "mlb_dict.pkl")  # adjust if different

# Optional fallback attribute names (human readable) — update if you have a canonical list
FALLBACK_ATTRIBUTES = [
    "PATTERN","COLOUR","SLEEVE_LENGTH","SLEEVE_TYPE","DRESS_LENGTH",
    "SIZE","NECK_LINE","FABRIC (OPTIONAL)","OCASSION"
]

device = torch.device("cpu")

def load_model(path: str):
    if not os.path.exists(path):
        logger.error("Model file not found at %s", path)
        raise FileNotFoundError(path)
    logger.info("Loading scripted model from %s", path)
    model = torch.jit.load(path, map_location=device)
    model.eval()
    logger.info("Loaded scripted model successfully.")
    return model

def load_mlb(path: str):
    if not os.path.exists(path):
        logger.warning("mlb file not found at %s", path)
        return None
    with open(path, "rb") as f:
        mlb = pickle.load(f)
    logger.info("Loaded mlb from %s", path)
    return mlb

# Load once
model = load_model(MODEL_PATH)
mlb = load_mlb(MLB_PATH)

# Preprocess transform (ImageNet style)
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "predictions": None})

def preprocess_image(file_bytes: bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # shape [1,3,224,224]
    return tensor, image

def extract_label_list(mlb_obj) -> Optional[List[str]]:
    """Return a list of label names if available from mlb object."""
    if mlb_obj is None:
        return None
    # sklearn MultiLabelBinarizer has attribute classes_
    classes = getattr(mlb_obj, "classes_", None) or getattr(mlb_obj, "classes", None)
    if classes is not None:
        try:
            return list(classes)
        except Exception:
            pass
    # Maybe mlb is saved as a plain list or dict
    if isinstance(mlb_obj, (list, tuple)):
        return list(mlb_obj)
    if isinstance(mlb_obj, dict) and "classes_" in mlb_obj:
        return list(mlb_obj["classes_"])
    # otherwise None
    return None

def get_label_names(model_output_len: int) -> (List[str], Optional[str]):
    """
    Try to get human-readable label names. Returns (labels, info_message).
    info_message contains a UI warning if shapes mismatch.
    """
    labels = extract_label_list(mlb)
    if labels:
        if len(labels) == model_output_len:
            logger.info("Using mlb.classes_ as label names.")
            return labels, None
        else:
            msg = (f"Warning: loaded label list length ({len(labels)}) "
                   f"does not match model output length ({model_output_len}). "
                   "Showing numeric indices with available labels where possible.")
            logger.warning(msg)
            # still return labels (caller will guard index access)
            return labels, msg

    # fallback attributes if they match exactly
    if len(FALLBACK_ATTRIBUTES) == model_output_len:
        logger.info("Using FALLBACK_ATTRIBUTES as label names.")
        return FALLBACK_ATTRIBUTES, None

    # final fallback numeric names
    numeric = [str(i) for i in range(model_output_len)]
    logger.info("Falling back to numeric label names.")
    return numeric, None

def predict_topk(img_tensor: torch.Tensor, topk: int = 10):
    """
    Runs the model, returns tuple: (top_predictions, features, info_message)
    top_predictions: list of (name, prob, index)
    features: list of floats or None
    info_message: optional string for UI
    """
    with torch.no_grad():
        logger.info("Running model forward pass")
        out = model(img_tensor)  # model might return tensor OR tuple/list like (logits, features)
        features = None
        # handle possible tuple outputs
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            logits = out[0]
            # try to get features if second item exists
            if len(out) >= 2:
                try:
                    features_tensor = out[1]
                    # convert to 1D numpy if batch dim used
                    features = features_tensor.squeeze(0).cpu().numpy().tolist()
                except Exception:
                    features = None
        else:
            logits = out

        logits = logits.cpu()
        probs = torch.sigmoid(logits).squeeze(0).numpy()  # [N]
        model_len = probs.shape[0]

        label_names, info_message = get_label_names(model_len)

        # pick topk
        idxs = probs.argsort()[::-1][:topk]

        top = []
        for i in idxs:
            # if label list shorter than index, use numeric string
            name = label_names[i] if i < len(label_names) else str(i)
            top.append((name, float(probs[i]), int(i)))

        # features: truncate for UI (first 128 dims) to avoid huge payloads
        if features is not None:
            features_preview = features[:128]
        else:
            features_preview = None

        return top, features_preview, info_message

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    logger.info("Received /predict request")
    contents = await file.read()
    img_tensor, pil_image = preprocess_image(contents)
    logger.info("Transformed image tensor shape: %s", str(img_tensor.shape))

    try:
        top_preds, features_preview, info_message = predict_topk(img_tensor, topk=12)
        logger.info("Inference complete")
    except Exception as e:
        logger.exception("Inference error: %s", e)
        top_preds = [("error", 0.0, -1)]
        features_preview = None
        info_message = f"Inference error: {e}"

    # Convert PIL image to base64 for embedding in template
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "predictions": top_preds,
        "image_b64": img_b64,
        "features": features_preview,
        "info_message": info_message,
        "presented_by": [
            "23061-CS-010 Naveed ullah khan",
            "23061-CS-018 M.Shiva datta.",
            "23061-CS-032 Ayesha Begum."
        ]
    })

@app.get("/health")
def health():
    return {"status": "ok"}
