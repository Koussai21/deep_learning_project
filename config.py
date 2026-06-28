import os

SEED = 42

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR    = os.path.join(BASE_DIR, "saved_models")
MLFLOW_URI    = os.path.join(BASE_DIR, "mlruns")

IMAGE_SIZE      = 224
NUM_CLASSES     = 14
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

BATCH_SIZE     = 32
NUM_EPOCHS     = 30
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.3
PATIENCE       = 7

LATENT_DIM     = 128
VAE_BETA       = 1.0
ANOMALY_THRESHOLD_PERCENTILE = 95

MAX_TEXT_LEN   = 128
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
FUSION_STRATEGY = "late"

EXPERIMENT_CLASSIFICATION = "chest_classification"
EXPERIMENT_ANOMALY        = "chest_anomaly_detection"
EXPERIMENT_MULTIMODAL     = "chest_multimodal"
