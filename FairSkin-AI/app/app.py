from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairskin_ai.checkpoint import load_checkpoint
from fairskin_ai.data import build_transforms, skin_group
from fairskin_ai.utils import resolve_device

st.set_page_config(page_title="FairSkin-AI", page_icon="🧪", layout="wide")
st.title("FairSkin-AI")
st.caption("Fairness-aware dermatology image classification — research prototype")
st.warning("Research/education only. This software is not a medical device and must not be used for diagnosis or treatment decisions.")

checkpoint = st.sidebar.text_input("Checkpoint", "artifacts/best_model.pt")
report_path = st.sidebar.text_input("Fairness report", "artifacts/fairness_report.json")
fitz = st.sidebar.selectbox("Fitzpatrick type (for subgroup context)", [1,2,3,4,5,6], index=2)

@st.cache_resource
def get_model(path: str):
    device = resolve_device("auto")
    model, payload = load_checkpoint(path, device)
    return model, payload, device

uploaded = st.file_uploader("Upload a dermatology image", type=["jpg", "jpeg", "png", "webp"])
if uploaded is not None:
    if not Path(checkpoint).exists():
        st.error(f"Checkpoint not found: {checkpoint}. Train a model first.")
        st.stop()
    image = Image.open(uploaded).convert("RGB")
    model, payload, device = get_model(checkpoint)
    transform = build_transforms(int(payload["image_size"]), False)
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    idx_to_class = {int(v): str(k) for k, v in payload["class_to_idx"].items()}
    pred_idx = int(probs.argmax())
    c1, c2 = st.columns([1,1])
    with c1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with c2:
        st.metric("Predicted category", idx_to_class[pred_idx])
        st.metric("Model confidence", f"{float(probs[pred_idx])*100:.1f}%")
        prob_df = pd.DataFrame({"class": [idx_to_class[i] for i in range(len(probs))], "probability": probs.numpy()})
        st.bar_chart(prob_df.set_index("class"))

    group = skin_group(fitz, "three_groups")
    st.subheader(f"Fairness context — skin group {group}")
    if Path(report_path).exists():
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        gm = report.get("groups", {}).get(group)
        if gm:
            cols = st.columns(4)
            cols[0].metric("Group n", int(gm["n"]))
            cols[1].metric("Accuracy", f"{gm['accuracy']:.3f}")
            cols[2].metric("Balanced accuracy", f"{gm['balanced_accuracy']:.3f}")
            cols[3].metric("Macro F1", f"{gm['macro_f1']:.3f}")
        else:
            st.info("No metrics were available for this subgroup in the saved evaluation report.")
    else:
        st.info("Train/evaluate the model to generate a fairness report.")
