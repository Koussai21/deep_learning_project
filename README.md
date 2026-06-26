# Système d'aide au tri radiologique — Projet Deep Learning

Système complet d'aide au tri radiologique sur radiographies thoraciques :
**classification multi-label supervisée**, **détection d'anomalies (AE/VAE)**,
**preuve de concept multimodale (image + texte)**, **tracking MLflow** et
**démonstrateur Streamlit**.

Ce dépôt répond au cahier des charges *Projet Deep Learning — Système d'aide au tri radiologique*.

---

## 1. Architecture du projet

```
deep_learning_project/
├── config.py                     # Hyperparamètres et chemins centralisés
├── requirements.txt
├── run_all_experiments.py        # Lance toute la campagne expérimentale
│
├── data/
│   ├── dataset.py                # ChestMNIST (multi-label, 14 pathologies)
│   └── multimodal_dataset.py     # OpenI (image + compte-rendu)
│
├── models/
│   ├── cnn_scratch.py            # CNN entraîné depuis zéro
│   ├── transfer_learning.py      # DenseNet121 / ResNet50 / EfficientNet
│   ├── vit.py                    # Vision Transformer + hybride CNN/Transformer
│   ├── autoencoder.py            # AE convolutionnel + VAE
│   └── multimodal.py             # Fusions précoce / intermédiaire / tardive
│
├── training/
│   ├── utils.py                  # Seed, early stopping, checkpoints
│   ├── train_classification.py   # Entraînement supervisé + MLflow
│   ├── train_anomaly.py          # Entraînement AE/VAE + MLflow
│   └── train_multimodal.py       # Entraînement multimodal + MLflow
│
├── evaluation/
│   └── metrics.py                # AUC, mAP, F1, courbes ROC, reconstructions
│
├── mlflow_utils/
│   └── tracking.py               # Helpers MLflow (runs, artefacts, modèles)
│
├── app/
│   └── streamlit_app.py          # Démonstrateur applicatif
│
└── notebooks/
    └── 01_eda.py                 # Analyse exploratoire + export d'échantillons
```

## 2. Installation

```bash
cd deep_learning_project
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

> **GPU recommandé.** Le code détecte automatiquement CUDA. Sur CPU, réduisez
> `IMAGE_SIZE` à 64 dans `config.py` et utilisez `--epochs 5` pour un test rapide.

## 3. Données

| Composante | Dataset | Téléchargement |
|-----------|---------|----------------|
| Classification & anomalies | **ChestMNIST** (MedMNIST) | automatique au 1er lancement |
| Multimodal image + texte | **OpenI** (Indiana University) | manuel — voir ci-dessous |

ChestMNIST est téléchargé automatiquement par `medmnist`. Pour OpenI, placez :
- les images dans `data/raw/openi/images/`
- les rapports XML dans `data/raw/openi/ecgen-radiology/`

(source : <https://openi.nlm.nih.gov/faq#collection>)

## 4. Utilisation

### 4.1 Analyse exploratoire
```bash
python notebooks/01_eda.py
```
Génère distribution des labels, co-occurrences, grille d'exemples et exporte
des radiographies PNG pour tester le démonstrateur.

### 4.2 Classification supervisée (3+ architectures)
```bash
python -m training.train_classification --model cnn_scratch
python -m training.train_classification --model densenet121
python -m training.train_classification --model vit
python -m training.train_classification --model hybrid
```

### 4.3 Détection d'anomalies
```bash
python -m training.train_anomaly --model ae
python -m training.train_anomaly --model vae
```

### 4.4 Multimodal (image / texte / fusion)
```bash
python -m training.train_multimodal --mode image
python -m training.train_multimodal --mode text
python -m training.train_multimodal --mode fusion --fusion late
```

### 4.5 Tout lancer
```bash
python run_all_experiments.py --epochs 10
```

### 4.6 Visualiser les expériences MLflow
```bash
mlflow ui --backend-store-uri ./mlruns
# puis ouvrir http://localhost:5000
```

### 4.7 Lancer le démonstrateur
```bash
streamlit run app/streamlit_app.py
```

## 5. Correspondance avec le cahier des charges

| Exigence | Implémentation |
|----------|----------------|
| 4.1 — 3 architectures supervisées | `cnn_scratch.py`, `transfer_learning.py`, `vit.py` |
| 4.2 — AE ou VAE | `autoencoder.py` (AE **et** VAE) |
| 4.3 — Multimodal image+texte, 3 fusions | `multimodal.py` (early / intermediate / late) |
| 4.4 — Tracking MLflow | `mlflow_utils/tracking.py`, intégré à tous les scripts |
| 4.5 — Démonstrateur | `app/streamlit_app.py` (Streamlit) |
| 4.6 — Pipeline reproductible | seed fixe, split train/val/test, early stopping, best-model |
| 6 — Régularisation/optimisation | dropout, weight decay, AdamW, scheduler cosine, augmentation, early stopping |

## 6. Choix techniques

- **Loss multi-label** : `BCEWithLogitsLoss` avec `pos_weight` calculé sur la
  fréquence des classes → compense le fort déséquilibre de ChestMNIST.
- **Métriques** : AUC-ROC (macro/micro), mAP, F1 — adaptées aux classes
  déséquilibrées plutôt que l'accuracy.
- **Anomalies** : entraînement sur images saines uniquement, seuil au 95ᵉ
  percentile de l'erreur de reconstruction, score d'atypicité au runtime.
- **Reproductibilité** : `set_seed()` fixe Python/NumPy/PyTorch + cudnn
  déterministe ; meilleur modèle sauvegardé et ré-exposé dans le démonstrateur.

## 7. Limites & avertissement

> ⚠️ **Prototype de recherche — usage non clinique.**
> ChestMNIST est une version basse résolution ; les labels Openly dérivés du
> texte par mots-clés sont faibles (proof-of-concept). Voir la section
> « Analyse critique » du rapport pour la discussion complète.

## 8. Livrables

1. **Code** — ce dépôt (entraînement supervisé, AE/VAE, multimodal, démonstrateur, MLflow).
2. **Démonstrateur** — `streamlit run app/streamlit_app.py`.
3. **MLflow** — runs et meilleur modèle dans `./mlruns` (`mlflow ui`).
4. **Rapport** — structure détaillée dans [`RAPPORT.md`](RAPPORT.md).
