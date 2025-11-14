# quantize.py
import os
from pathlib import Path
import io
import h5py
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path.cwd()
H5_PATH = ROOT / "model" / "finetuned_resnet50_multilabel.h5"
MLB_PATH = ROOT / "model" / "mlb_dict.pkl"
OUT_DIR = ROOT / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_QUANT_STATE = OUT_DIR / "model_quantized_state.pth"
OUT_SCRIPTED = OUT_DIR / "model_quantized_scripted.pt"
OUT_ZIP = OUT_DIR / "model_quantized_scripted.zip"

def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T']:
        if abs(num) < 1024.0:
            return f"{num:3.2f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f}P{suffix}"

if not H5_PATH.exists():
    raise FileNotFoundError(f"H5 not found: {H5_PATH}")
if not MLB_PATH.exists():
    raise FileNotFoundError(f"mlb pickle not found: {MLB_PATH}")

with open(MLB_PATH, "rb") as f:
    mlb_dict = pickle.load(f)
target_cols = list(mlb_dict.keys())
attr_sizes = [len(mlb_dict[c].classes_) for c in target_cols]

def load_model_h5_to_state_dict(h5_path: Path):
    st = {}
    with h5py.File(str(h5_path), "r") as h5f:
        if "state_dict" in h5f:
            grp = h5f["state_dict"]
        elif "model" in h5f:
            grp = h5f["model"]
        else:
            first = list(h5f.keys())[0]
            grp = h5f[first]
        for ds in grp:
            key = ds.replace("__", "/")
            arr = grp[ds][()]
            arr = np.array(arr)
            st[key] = torch.tensor(arr)
    return st

print("Loading state dict from HDF5...")
state_dict = load_model_h5_to_state_dict(H5_PATH)

class MultiHeadResNet(nn.Module):
    def __init__(self, backbone: nn.Module, attr_sizes, in_features: int):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(in_features, s) for s in attr_sizes])

    def forward(self, x):
        feats = self.backbone(x)
        return [head(feats) for head in self.heads]

print("Building model architecture...")
backbone = models.resnet50(weights=None)
in_features = backbone.fc.in_features
backbone.fc = nn.Identity()
model = MultiHeadResNet(backbone, attr_sizes, in_features)

print("Loading weights (strict=False)...")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))

model.eval()

print("Applying dynamic quantization to nn.Linear modules...")
quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# Save state dict of quantized model (smaller than original state in many cases)
torch.save(quantized.state_dict(), OUT_QUANT_STATE)
print("Saved quantized state dict to:", OUT_QUANT_STATE)

# Script the quantized model (traces forward with example input)
example_input = torch.randn(1, 3, 224, 224)
try:
    scripted = torch.jit.trace(quantized, example_input, strict=False)
    scripted.save(OUT_SCRIPTED)
    print("Saved scripted quantized model to:", OUT_SCRIPTED)
except Exception as e:
    print("Warning: scripting/tracing failed:", e)

# Optionally zip the scripted model
if OUT_SCRIPTED.exists():
    import zipfile
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_SCRIPTED, arcname=OUT_SCRIPTED.name)
    print("Zipped scripted model to:", OUT_ZIP)

# Report sizes
for p in [H5_PATH, OUT_QUANT_STATE, OUT_SCRIPTED, OUT_ZIP]:
    if p.exists():
        print(f"{p.name}: {sizeof_fmt(p.stat().st_size)}")
    else:
        print(f"{p.name}: MISSING")
