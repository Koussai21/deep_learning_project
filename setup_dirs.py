import os
import config

dirs = [
    config.DATA_DIR,
    config.MODELS_DIR,
    os.path.join(config.BASE_DIR, "mlruns"),
    os.path.join(config.BASE_DIR, "notebooks", "eda_outputs"),
    os.path.join(config.BASE_DIR, "notebooks", "eda_outputs", "samples"),
    os.path.join(config.DATA_DIR, "openi", "images"),
    os.path.join(config.DATA_DIR, "openi", "ecgen-radiology"),
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"  OK  {d}")

print("\nAll directories ready.")
