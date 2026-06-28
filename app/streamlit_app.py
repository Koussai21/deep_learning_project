import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.cnn_scratch import CNNFromScratch
from models.transfer_learning import TransferModel
from models.vit import ViTClassifier, HybridCNNViT
from models.autoencoder import ConvAE, VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model registry ────────────────────────────────────────────────────────────
# ViT positional embeddings are fixed at the size used during training (128px).
# Feeding a different resolution at inference raises an AssertionError in timm.
# CNN scratch and Hybrid are also trained at 128px in the optimal config.
MODEL_IMAGE_SIZES = {
    "cnn_scratch":     128,
    "densenet121":     224,
    "resnet50":        224,
    "efficientnet_b0": 224,
    "vit":             128,
    "hybrid":          128,
}

CLASSIFIER_BUILDERS = {
    "cnn_scratch":     CNNFromScratch,
    "densenet121":     lambda: TransferModel("densenet121"),
    "resnet50":        lambda: TransferModel("resnet50"),
    "efficientnet_b0": lambda: TransferModel("efficientnet_b0"),
    "vit":             lambda: ViTClassifier(img_size=128),
    "hybrid":          HybridCNNViT,
}


@st.cache_resource
def load_classifier(name: str):
    path = os.path.join(config.MODELS_DIR, f"classifier_{name}.pt")
    if not os.path.exists(path):
        return None
    model = CLASSIFIER_BUILDERS[name]()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_anomaly(name: str):
    path = os.path.join(config.MODELS_DIR, f"anomaly_{name}.pt")
    thr_path = os.path.join(config.MODELS_DIR, f"anomaly_{name}_threshold.npy")
    if not os.path.exists(path):
        return None, None
    model = VAE(image_size=64) if name == "vae" else ConvAE(image_size=64)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    threshold = float(np.load(thr_path)[0]) if os.path.exists(thr_path) else None
    return model, threshold


def preprocess(image: Image.Image, size: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(image.convert("RGB")).unsqueeze(0)


def preprocess_anomaly(image: Image.Image, size: int = 64) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(image.convert("RGB")).unsqueeze(0)


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Radiological Triage Assistant", layout="wide")
st.title("🩻 Radiological Triage Assistant")
st.caption("Multi-label pathology prediction · Anomaly detection · Multimodal proof-of-concept")

with st.sidebar:
    st.header("Configuration")
    classifier_name = st.selectbox("Classification model", list(CLASSIFIER_BUILDERS.keys()), index=1)
    anomaly_name    = st.selectbox("Anomaly model", ["vae", "ae"])
    threshold_prob  = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.caption("⚠️ Research prototype — not for clinical use.")

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])

col_left, col_right = st.columns([1, 1.3])

if uploaded is not None:
    image = Image.open(uploaded)
    with col_left:
        st.image(image, caption="Input radiograph", use_column_width=True)

    with col_right:
        # ── Supervised classification ─────────────────────────────────────
        st.subheader("1 · Supervised pathology predictions")
        clf = load_classifier(classifier_name)
        if clf is None:
            st.warning(f"No trained checkpoint for `{classifier_name}`. "
                       f"Run `python -m training.train_classification --model {classifier_name}`.")
        else:
            img_size = MODEL_IMAGE_SIZES.get(classifier_name, config.IMAGE_SIZE)
            x = preprocess(image, img_size).to(DEVICE)
            with torch.no_grad():
                probs = torch.sigmoid(clf(x)).cpu().numpy()[0]

            results = sorted(zip(config.CLASS_NAMES, probs), key=lambda t: -t[1])
            for name, p in results:
                flagged = p >= threshold_prob
                st.progress(float(p), text=f"{'🔴' if flagged else '⚪'} {name}: {p:.1%}")

            positives = [n for n, p in results if p >= threshold_prob]
            if positives:
                st.error("Flagged pathologies: " + ", ".join(positives))
            else:
                st.success("No pathology above threshold.")

        # ── Anomaly detection ─────────────────────────────────────────────
        st.subheader("2 · Anomaly / out-of-distribution score")
        ae_model, threshold = load_anomaly(anomaly_name)
        if ae_model is None:
            st.warning(f"No trained checkpoint for `{anomaly_name}`. "
                       f"Run `python -m training.train_anomaly --model {anomaly_name}`.")
        else:
            xa = preprocess_anomaly(image).to(DEVICE)
            with torch.no_grad():
                out = ae_model(xa)
                x_hat = out[0]
                score = ae_model.anomaly_score(xa, x_hat).item()

            st.metric("Reconstruction error", f"{score:.5f}")
            if threshold is not None:
                if score > threshold:
                    st.error(f"⚠️ Atypical image (score > threshold {threshold:.5f})")
                else:
                    st.success(f"✓ Within normal distribution (threshold {threshold:.5f})")

            # show reconstruction
            recon_img = x_hat[0].cpu().numpy().transpose(1, 2, 0)
            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
            st.image(recon_img, caption="Reconstruction", width=200)

    # ── Multimodal (optional) ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3 · Multimodal — process a radiology report (optional)")
    report = st.text_area("Paste a radiology report (FINDINGS / IMPRESSION)", height=120)
    if report.strip():
        from training.train_multimodal import derive_labels
        weak = derive_labels(report)
        mentioned = [config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES) if weak[i] > 0]
        if mentioned:
            st.info("Pathologies mentioned in report: " + ", ".join(mentioned))
        else:
            st.info("No known pathology keyword detected in the report.")
        st.caption("Late fusion combines image probabilities with report evidence. "
                   "Train with `python -m training.train_multimodal --mode fusion --fusion late`.")
else:
    st.info("⬆️ Upload a chest X-ray to start. "
            "ChestMNIST test images can be exported with `notebooks/01_eda.py`.")
