import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR    = os.path.join(BASE_DIR, "saved_models")
MLFLOW_URI    = os.path.join(BASE_DIR, "mlruns")

# ── Dataset ───────────────────────────────────────────────────────────────────
IMAGE_SIZE      = 224          # 64 | 128 | 224
NUM_CLASSES     = 14
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
NUM_EPOCHS     = 30
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.3
PATIENCE       = 7             # early stopping

# ── Anomaly detection ─────────────────────────────────────────────────────────
LATENT_DIM     = 128
VAE_BETA       = 1.0           # weight of KL term
ANOMALY_THRESHOLD_PERCENTILE = 95   # percentile on normal train scores

# ── Multimodal ────────────────────────────────────────────────────────────────
MAX_TEXT_LEN   = 128
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
FUSION_STRATEGY = "late"       # "early" | "intermediate" | "late"

# ── MLflow ────────────────────────────────────────────────────────────────────
EXPERIMENT_CLASSIFICATION = "chest_classification"
EXPERIMENT_ANOMALY        = "chest_anomaly_detection"
EXPERIMENT_MULTIMODAL     = "chest_multimodal"
