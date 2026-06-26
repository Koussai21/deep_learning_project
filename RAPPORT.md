# Rapport — Système d'aide au tri radiologique

> Trame du rapport final suivant la structure imposée (section 5 du cahier des
> charges). Les emplacements `[À COMPLÉTER]` doivent être remplis avec vos
> résultats expérimentaux réels (captures MLflow, chiffres, figures).

---

## 1. Problème
- **Contexte** : le volume de radiographies thoraciques dépasse la capacité de
  lecture des radiologues ; un système de tri automatique peut prioriser les cas urgents.
- **Objectif métier** : pré-trier les radiographies, signaler les pathologies
  probables et isoler les cas atypiques nécessitant une attention humaine.
- **Problématique IA** : classification multi-label déséquilibrée + détection
  d'anomalies hors distribution + exploitation du compte-rendu textuel.
- **Intérêt du tri radiologique** : réduction du délai diagnostique, aide à la
  priorisation, filet de sécurité pour les cas rares.

## 2. Données
- **Datasets** : ChestMNIST (principal, 14 pathologies multi-label), OpenI (multimodal image+texte).
- **Sources** : MedMNIST v2 ; Indiana University Chest X-ray (OpenI).
- **Structure** : ChestMNIST — images 28/64/128/224, labels multi-hot (14). OpenI — images PNG + rapports XML (FINDINGS/IMPRESSION).
- **Limites** : ChestMNIST basse résolution ; labels OpenI dérivés du texte (faibles).
- **Contraintes d'accès** : ChestMNIST libre ; MIMIC-CXR (alternative) sous accès restreint — OpenI retenu pour la faisabilité.
- **Justification du choix multimodal** : OpenI fournit de vrais comptes-rendus, contrairement à NIH ChestX-ray14 dont les textes ne sont pas exploitables.

## 3. Analyse exploratoire
*(figures générées par `notebooks/01_eda.py`)*
- **Distribution des labels** : `eda_outputs/label_distribution.png` — [commenter le déséquilibre].
- **Déséquilibre** : [% de chaque classe, % de « no finding »].
- **Exemples visuels** : `eda_outputs/sample_grid.png`.
- **Co-occurrences** : `eda_outputs/cooccurrence.png` — [paires fréquentes, ex. Effusion+Infiltration].
- **Premiers constats texte** : [longueur des rapports, vocabulaire dominant].

## 4. Préparation
- **Preprocessing images** : resize, conversion RGB, normalisation ImageNet.
- **Augmentation** : flip horizontal, rotation ±10°, jitter de luminosité/contraste (train uniquement).
- **Labels multi-label** : vecteurs multi-hot float, activation sigmoïde + BCE.
- **Texte** : tokenisation Bio_ClinicalBERT, troncature à 128 tokens.
- **Déséquilibre** : `pos_weight` dans la BCE (cf. `data/dataset.py::get_class_weights`).
- **Stratégie anti-fuite** : split train/val/test fixe (seed 42) ; [préciser le split par patient si identifiants disponibles].

## 5. Modélisation supervisée
Trois architectures comparées (cf. `models/`) :
1. **CNN from scratch** — 4 blocs Conv-BN-ReLU-MaxPool, GAP, tête FC. Justification : convolutions = extraction de features locales, BN = stabilisation, pooling = invariance spatiale.
2. **Transfer learning (DenseNet121)** — connexions denses, poids ImageNet, tête remplacée. Justification : réutilisation de features génériques, connexions résiduelles/denses contre le vanishing gradient.
3. **ViT / hybride CNN-Transformer** — patchs 16×16, self-attention, token [CLS]. Justification : capture du contexte global ; discussion de l'intérêt du ViT sur petit dataset médical.

| Modèle | Params | AUC macro (test) | mAP | F1 macro |
|--------|--------|------------------|-----|----------|
| CNN scratch | [À COMPLÉTER] | | | |
| DenseNet121 | | | | |
| ViT | | | | |
| Hybride | | | | |

## 6. Détection d'anomalies
- **Modèles** : AE convolutionnel et VAE (`models/autoencoder.py`).
- **Fonction de perte** : MSE (AE) ; ELBO = MSE + β·KL (VAE).
- **Protocole** : entraînement sur images saines uniquement ; seuil = 95ᵉ percentile de l'erreur de reconstruction sur le train normal.
- **Score** : erreur de reconstruction par échantillon.
- **Analyse des cas atypiques** : [exemples d'images au score élevé, figure `reconstruction_*.png`].
- **AUC normal vs pathologique** : [À COMPLÉTER].
- **Limites d'interprétation clinique** : un score élevé signale une atypie, pas un diagnostic.

## 7. Modélisation multimodale
- **Représentations** : image (EfficientNet-B0) + texte ([CLS] Bio_ClinicalBERT).
- **Fusion** : comparaison early / intermediate (cross-attention) / late.
- **Comparaison** :

| Modèle | AUC macro | mAP | F1 |
|--------|-----------|-----|-----|
| Image seule | [À COMPLÉTER] | | |
| Texte seul | | | |
| Fusion (late) | | | |

- **Alignement image-texte** : [discussion].
- **Modalités manquantes** : la fusion tardive dégrade gracieusement (moyenne sur modalités disponibles).

## 8. Evaluation
- **Métriques globales** : AUC macro/micro, mAP, F1, Hamming loss.
- **Par classe** : AUC par pathologie (cf. MLflow `auc_<classe>`).
- **Courbes** : ROC par pathologie (`roc_*.png`).
- **Matrices** : matrices de confusion multi-label.
- **Analyse critique** : [classes mal prédites, effet du déséquilibre].

## 9. Tracking MLflow
- **Organisation des runs** : 3 expériences (`chest_classification`, `chest_anomaly_detection`, `chest_multimodal`).
- **Paramètres testés** : modèle, LR, batch size, image size, β (VAE), stratégie de fusion.
- **Meilleur run** : [À COMPLÉTER — id du run, métrique].
- **Preuve du modèle déployé** : checkpoint loggé + modèle PyTorch enregistré, ré-exposé dans le démonstrateur.

## 10. Démonstrateur
- **Architecture** : Streamlit (`app/streamlit_app.py`).
- **Fonctionnalités** : upload radiographie → prédictions supervisées + score d'anomalie + reconstruction + traitement optionnel d'un compte-rendu.
- **Limites** : usage non clinique, dépend des checkpoints entraînés.
- **Interaction utilisateur** : choix du modèle, seuil de décision réglable.

## 11. Analyse critique
- **Robustesse** : [sensibilité au bruit, à la résolution].
- **Généralisation** : [écart train/test, transfert ChestMNIST → OpenI].
- **Erreurs fréquentes** : [classes rares, co-occurrences].
- **Coût calculatoire** : [temps d'entraînement par modèle, cf. `epoch_time_s` MLflow].
- **Apport multimodalité & AE/VAE** : [gain mesuré de la fusion ; utilité du score d'anomalie].

## 12. Conclusion et perspectives
- **Bilan** : [synthèse des performances].
- **Perspectives** : MIMIC-CXR pour un vrai multimodal, résolution 224, attention explicable (Grad-CAM), calibration des probabilités, validation clinique.

---

### Configuration matérielle (exigence section 6)
- **Matériel** : [GPU/CPU, RAM].
- **Temps d'entraînement** : [par modèle].
- **Contraintes de calcul** : [batch size, résolution retenue].

### Choix de régularisation et optimisation (exigence section 6)
- Augmentation : flip, rotation, color jitter.
- Early stopping : patience 7 sur l'AUC de validation.
- Dropout : 0.3 ; weight decay : 1e-4.
- Optimiseur : AdamW ; scheduler : CosineAnnealingLR ; batch size : 32.
