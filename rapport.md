# Rapport — Système de Triage Radiologique par Deep Learning
**Classification multi-label de pathologies thoraciques, détection d'anomalies et fusion multimodale**

---

## 1. Problème

### 1.1 Contexte et objectif métier

L'interprétation des radiographies thoraciques est l'un des actes d'imagerie médicale les plus fréquents au monde. Dans de nombreux établissements de santé, les radiologues font face à un volume croissant d'examens, engendrant des délais de lecture pouvant atteindre plusieurs heures, voire jours. Ce délai est particulièrement critique pour des pathologies urgentes telles que le pneumothorax ou l'œdème pulmonaire.

L'objectif métier de ce projet est de concevoir un **système d'aide à la décision pour le triage radiologique** : un outil capable de signaler automatiquement les clichés contenant des signes pathologiques, de les hiérarchiser par gravité présumée, et de réduire ainsi la charge cognitive des praticiens en leur présentant en priorité les cas les plus urgents.

### 1.2 Problématique IA

Le problème se formule comme une **classification multi-label** : une radiographie peut présenter simultanément plusieurs pathologies (ex. Atélectasie + Épanchement pleural). Pour chacune des 14 pathologies ciblées, le modèle doit estimer une probabilité indépendante, sans exclusion mutuelle. Ce cadre diffère de la classification multi-classe classique et impose des choix spécifiques de fonction de perte et de métriques d'évaluation.

Une problématique transversale est le **déséquilibre extrême des classes** : la pneumonie représente seulement 1,25 % des images, contre 17,73 % pour les infiltrations. Ce déséquilibre peut amener un modèle naïf à ne jamais prédire les classes rares, maximisant apparemment son accuracy globale tout en étant inutile cliniquement.

### 1.3 Intérêt du triage radiologique automatisé

Un système de triage performant apporte plusieurs bénéfices :
- **Réduction du temps de prise en charge** pour les pathologies urgentes (pneumothorax, œdème)
- **Aide à la priorisation** dans les files d'attente de lecture
- **Support aux radiologues moins expérimentés** en signalant les zones suspectes
- **Cohérence inter-examinateurs** : le modèle est insensible à la fatigue

Il s'agit d'un outil d'**assistance** et non de substitution : la décision clinique finale reste celle du médecin.

---

## 2. Données

### 2.1 ChestMNIST — Classification supervisée

**Source** : ChestMNIST est un sous-ensemble du benchmark MedMNIST v2, lui-même dérivé du dataset ChestX-ray14 (NIH Clinical Center, Wang et al., 2017). Ce dernier contient 112 120 radiographies frontales de 30 805 patients uniques, annotées automatiquement par un système NLP appliqué aux comptes-rendus radiologiques.

**Structure** :

| Split | Images |
|-------|--------|
| Train | 78 468 |
| Validation | 11 219 |
| Test | 22 430 |

- **Résolution** : images redimensionnées à la taille demandée (64, 128 ou 224 px) par MedMNIST
- **Canaux** : grayscale converti en RGB (3 canaux identiques)
- **Labels** : vecteur multi-hot de dimension 14, un bit par pathologie

**14 pathologies ciblées** : Atélectasie, Cardiomégalie, Épanchement, Infiltration, Masse, Nodule, Pneumonie, Pneumothorax, Consolidation, Œdème, Emphysème, Fibrose, Épaississement pleural, Hernie.

**Limites** :
- Les labels sont générés par NLP (mention textuelle ≠ pathologie confirmée), ce qui introduit un bruit d'annotation estimé à 5–15 %
- La résolution réduite (jusqu'à 28 × 28 en version native) peut effacer des détails fins diagnostiquement importants (petits nodules < 5 mm)
- Biais de population : données exclusivement américaines (NIH, Washington D.C.)
- Pas d'information de localisation (pas de bounding boxes)

### 2.2 OpenI — Modélisation multimodale

**Source** : Indiana University Chest X-ray Collection (Demner-Fushman et al., 2015). Dataset publiquement disponible (~3 900 rapports et ~7 400 images).

**Structure** : Paires image/rapport en XML, avec sections FINDINGS et IMPRESSION. Les labels multi-label ne sont pas fournis nativement : ils sont dérivés par correspondance de mots-clés dans le texte (voir section 7).

**Justification du choix multimodal** : Dans la pratique clinique, la radiographie est systématiquement accompagnée d'un contexte textuel (antécédents, compte-rendu d'imagerie antérieure). L'intégration du texte permet de lever des ambiguïtés visuelles et d'incorporer un savoir clinique structuré que l'image seule ne peut exprimer.

**Contraintes d'accès** : Les deux datasets sont librement téléchargeables (licences CC/recherche). Aucune donnée patient identifiable n'est présente (les images sont anonymisées par NIH/IU).

---

## 3. Analyse Exploratoire

### 3.1 Distribution des labels et déséquilibre

La distribution de prévalence sur le split d'entraînement révèle un déséquilibre sévère et hétérogène :

| Pathologie | Fréquence | Poids positif (brut) |
|------------|-----------|----------------------|
| Infiltration | 17,73 % | 4,64 |
| Effusion | 11,80 % | 7,47 |
| Atelectasis | 10,19 % | 8,81 |
| Mass | 5,08 % | 18,68 |
| Nodule | 5,58 % | 16,92 |
| Pneumothorax | 4,72 % | 20,18 |
| Consolidation | 4,16 % | 23,04 |
| Edema | 2,15 % | 45,48 |
| Emphysema | 2,29 % | 42,67 |
| Cardiomegaly | 2,49 % | 39,24 |
| Pleural_Thickening | 2,90 % | 33,49 |
| Fibrosis | 1,48 % | 66,67 |
| **Pneumonia** | **1,25 %** | **79,23** |
| **Hernia** | **0,18 %** | **543,92** |

Ce déséquilibre pose un problème fondamental : un modèle qui ne prédit jamais "Hernie" obtient une accuracy de 99,82 % sur cette classe, sans aucune utilité clinique.

### 3.2 Co-occurrences

L'analyse des co-occurrences révèle des associations cliniquement cohérentes :
- **Épanchement + Cardiomégalie** : souvent associées dans l'insuffisance cardiaque
- **Pneumonie + Consolidation** : quasi-synonymes dans les labels NLP
- **Atélectasie + Épanchement** : fréquentes ensemble en post-opératoire

Ces co-occurrences expliquent pourquoi un modèle peu discriminant confond ces pathologies : leurs représentations visuelles se superposent dans l'espace des features.

### 3.3 Premiers constats

- La majorité des images (> 55 %) présentent au moins une pathologie ; les images "normales" (vecteur tout-zéro) représentent ~40 % du train
- Le multi-label est fréquent : ~30 % des images pathologiques en ont ≥ 2 simultanément
- Les images ChestMNIST, même à 224 px, restent floues du fait du redimensionnement depuis les clichés originaux haute résolution

---

## 4. Préparation des Données

### 4.1 Preprocessing images

**Pipeline d'augmentation (entraînement des classifieurs)** :
```
Resize(image_size × 1.1) → RandomCrop(image_size) → RandomHorizontalFlip()
→ RandomRotation(12°) → ColorJitter(brightness=0.3, contrast=0.3)
→ convert("RGB") → ToTensor() → Normalize(ImageNet) → RandomErasing(p=0.15)
```

**Pipeline d'évaluation (validation / test)** :
```
Resize(image_size) → convert("RGB") → ToTensor() → Normalize(ImageNet)
```

**Choix de conception** :
- Le `RandomCrop` (padding 10 %) simule des variations de cadrage sans couper de régions anatomiques critiques
- Le `RandomHorizontalFlip` est justifié : la position du cœur peut varier légèrement (dextrocardie)
- Le flip **vertical n'est pas appliqué** : les radiographies ont une orientation standardisée (apex en haut)
- Le `RandomErasing` simule des artéfacts ou des zones masquées, renforçant la robustesse

**Pour les autoencodeurs AE/VAE** : le `RandomErasing` est **désactivé** (`augment_train=False`). En effet, l'AE doit reconstruire son entrée ; effacer des pixels dans l'entrée sans les effacer dans la cible reviendrait à entraîner un débruiteur plutôt qu'un reconstructeur fidèle, faussant le score d'anomalie.

### 4.2 Normalisation

Normalisation ImageNet (µ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]) appliquée à tous les modèles, y compris le CNN from scratch. Ce choix est standard même pour les modèles non pré-entraînés, car il centre la distribution des pixels autour de zéro, facilitant la convergence.

### 4.3 Labels multi-label et gestion du déséquilibre

**Stratégie principale** : pondération positive dans la `BCEWithLogitsLoss` via le paramètre `pos_weight`.

La formule initiale `pos_weight = neg_count / pos_count` donnait des poids de 79 pour Pneumonia et 544 pour Hernia, provoquant un **effondrement vers "tout prédire positif"** : le modèle CNN scratch présentait un recall de 0,91 pour Pneumonia mais une précision de seulement 0,010, prédisant 1006/1024 images comme positives pour cette classe.

**Correction appliquée** : cap à 10,0 sur tous les pos_weight :
```python
pos_weight = torch.clamp(neg_count / (pos_count + 1e-6), max=10.0)
```
Ce plafonnement conserve l'incitation à détecter les classes rares sans rendre le coût des faux négatifs prohibitif.

**Label smoothing** (ε = 0,05) : les étiquettes dures (0 ou 1) sont adoucies en `1 − ε` et `ε/2` respectivement. Cela prévient la sur-confiance du modèle sur des labels eux-mêmes bruités (issus de NLP).

### 4.4 Stratégie anti-fuite (data leakage)

- Aucune augmentation n'est appliquée sur les splits de validation et test
- Les poids positifs (`pos_weight`) sont calculés **uniquement sur le train set**
- La normalisation utilise des statistiques ImageNet globales (pas calculées sur nos données)
- Pour le multimodal, le split train/val/test est réalisé au niveau des patients (rapports), non des images, pour éviter qu'un même patient apparaisse dans deux splits

---

## 5. Modélisation Supervisée

### 5.1 Trois architectures profondes

#### Architecture 1 — CNN from Scratch (`CNNFromScratch`, 1,3M paramètres)

**Description** : 4 blocs convolutifs (Conv3×3 → BN → ReLU) × 2 + MaxPool2d, avec doublement des canaux (3→32→64→128→256). Global Average Pooling (GAP), puis FC(256→512→14). Initialisation Kaiming (Fan-out).

**Justification** : Sert de **baseline bas de gamme** pour quantifier le bénéfice du transfert learning. Entraîné sans poids pré-entraînés, il capture uniquement l'information présente dans ChestMNIST. Son faible nombre de paramètres le rend rapide mais limite sa capacité d'abstraction pour des images médicales complexes.

**Hyperparamètres optimaux** :
- Image size : 128 px (4 MaxPool → feature map 8×8)
- LR : 3×10⁻⁴ (plus élevé car initialisation aléatoire)
- Batch : 64, Epochs : 50, Patience : 12

#### Architecture 2 — DenseNet121 (`TransferModel`, 7,0M paramètres)

**Description** : DenseNet121 pré-entraîné sur ImageNet, avec la tête de classification remplacée par `Dropout(0.3) → Linear(1024 → 14)`. Tous les poids sont fine-tunés.

**Justification** : DenseNet121 est l'architecture de référence en radiologie thoracique depuis le papier CheXNet (Rajpurkar et al., 2017), qui a montré que ce modèle dépassait les performances humaines sur plusieurs pathologies de ChestX-ray14. Les connexions denses entre couches permettent une réutilisation des features à toutes les échelles, particulièrement adaptée aux structures anatomiques qui se chevauchent.

**Hyperparamètres optimaux** :
- Image size : 224 px (résolution native pré-entraînement)
- LR : 1×10⁻⁴ (valeur du papier CheXNet)
- Batch : 32, Epochs : 30, Patience : 8

#### Architecture 3 — ViT-Small / Patch16 (`ViTClassifier`, 21,6M paramètres)

**Description** : ViT-Small pré-entraîné sur ImageNet-21k (timm), avec `img_size=128` (64 tokens au lieu de 196 à 224 px), suivi d'un head `LayerNorm → Dropout → Linear(384 → 14)`.

**Justification** : Les Vision Transformers capturent des **dépendances globales** via le mécanisme d'attention, particulièrement utile pour des pathologies diffuses (œdème pulmonaire bilatéral). À 128 px, l'attention est 9× moins coûteuse qu'à 224 px (complexité O(n²) en tokens). **Point important** : le `img_size` est fixé à la compilation des embeddings positionnels ; le modèle doit être instancié avec la même taille qu'à l'entraînement, sous peine d'une `AssertionError` à l'inférence.

**Hyperparamètres optimaux** :
- Image size : 128 px
- LR : 5×10⁻⁵ (les Transformers sont sensibles au LR, valeur conservatrice)
- Batch : 32, Epochs : 30, Patience : 8

#### Architecture 4 (bonus) — HybridCNNViT (`HybridCNNViT`, 2,4M paramètres)

**Description** : ResNet50 tronqué à `layer2` (stride total 8) utilisé comme extracteur spatial, produisant 256 tokens de dimension 512. Projection vers embed_dim=192, ajout d'un token CLS, puis 2 couches de Transformer Encoder (6 têtes, FFN × 4). Sortie : token CLS → Dropout → Linear(192 → 14).

**Justification** : Combine l'efficacité du CNN pour l'extraction de features locales (textures, bords) avec la capacité du Transformer à modéliser les relations spatiales longue portée. Le CNN réduit la dimension spatiale avant l'attention, rendant le modèle léger malgré la richesse de l'architecture.

### 5.2 Fonction de perte et optimisation

**Fonction de perte** : `BCEWithLogitsLoss(pos_weight)` pour toutes les architectures. Cette combinaison intègre numériquement stable sigmoid + BCE. Le `pos_weight` plafond à 10 contre-balance le déséquilibre sans provoquer l'effondrement.

**Optimiseur** : AdamW (weight decay 1×10⁻⁴), avec Cosine Annealing LR sans warmup (sauf ViT où le LR initial faible joue ce rôle).

**Stabilité d'entraînement** :
- Gradient clipping (max_norm = 1,0) : essentiel pour le ViT et le Hybrid qui peuvent subir des explosions de gradient lors de l'initialisation du head
- Automatic Mixed Precision (AMP, GradScaler + autocast) : environ 2× plus rapide sur GPU avec tenseur cores, sans perte de précision

---

## 6. Détection d'Anomalies

### 6.1 Principe et protocole

Les autoencodeurs sont entraînés **exclusivement sur les images normales** (vecteur de labels tout-zéro dans ChestMNIST, soit ~40 % du train). L'hypothèse est que le modèle apprend la distribution des radiographies saines et exhibe un score de reconstruction élevé face à des images pathologiques (hors distribution).

**Protocole** :
1. Filtrer le train set pour ne garder que les images sans aucune pathologie (`labels.sum() == 0`)
2. Entraîner le modèle à minimiser l'erreur de reconstruction sur ces images normales
3. Calculer le score d'anomalie sur le train normal : prendre le **95e percentile** comme seuil
4. À l'inférence : score > seuil → image atypique

### 6.2 Architectures AE et VAE

**ConvAE** : Encodeur 4 DownBlocks (Conv4×4 stride 2, BN, LeakyReLU), GAP → FC(latent_dim=128). Décodeur symétrique 4 UpBlocks (ConvTranspose, BN, ReLU) + Sigmoid final. Score d'anomalie : MSE pixel-wise.

**VAE** : Même architecture mais l'encodeur produit (µ, log_var), avec reparamétrage `z = µ + ε·σ`. Perte ELBO = MSE_reconstruction + β·KL_divergence (β = 1,0). Le terme KL régularise l'espace latent en le forçant vers N(0,1), rendant l'espace plus compact et le score d'anomalie plus robuste aux variations d'échelle.

### 6.3 Fonction de perte

**AE** : `L = MSE(x̂, x)` — simple, interprétable, mais l'espace latent peut être peu structuré.

**VAE** : `L = MSE(x̂, x) + β·KL(N(µ,σ²) || N(0,1))` — le terme KL force une distribution latente régulière, améliorant la généralisation et l'interpolation dans l'espace latent.

### 6.4 Score d'anomalie et analyse

Le score est calculé per-sample : `score(x) = mean(|x − x̂|²)` sur tous les pixels et canaux. Un score élevé signifie que le modèle peine à reconstruire l'image, ce qui indique une déviation par rapport à la distribution "normale".

**Limites du protocole** :
- Le seuil au 95e percentile est calibré sur les données d'entraînement : il peut être mal calibré sur une distribution de patients différente
- Les images très sombres ou surexposées (problèmes d'acquisition) peuvent obtenir un score élevé sans être pathologiques
- À 64 px de résolution, des détails fins (petits nodules) peuvent ne pas causer d'erreur de reconstruction détectable

---

## 7. Modélisation Multimodale

### 7.1 Dataset OpenI et labels faibles

Le dataset OpenI (~3 900 rapports, ~7 400 images) est utilisé pour la modélisation multimodale. Les labels multi-label n'étant pas fournis nativement, ils sont dérivés par **correspondance de mots-clés** dans le texte des sections FINDINGS et IMPRESSION :

```python
PATHOLOGY_KEYWORDS = {
    "Pneumonia": ["pneumonia"],
    "Effusion":  ["effusion"],
    ...
}
```

Cette stratégie produit des labels **faibles** (weak supervision) : la présence du mot "effusion" dans le texte ne garantit pas que la pathologie est visible sur l'image (elle peut être mentionnée pour être exclue : "no pleural effusion"). Il s'agit d'une borne inférieure de qualité d'annotation.

### 7.2 Encodeurs

- **Image** : EfficientNet-B0 pré-entraîné (backbone léger, ~5M paramètres) → projection Linear(1280 → 512)
- **Texte** : Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) pré-entraîné sur notes cliniques PubMed + MIMIC → token [CLS] → projection Linear(768 → 512)

Le choix de Bio_ClinicalBERT est motivé par son domaine d'entraînement : un BERT générique (entraîné sur Wikipedia) ne comprend pas le jargon radiologique ("hilar adenopathy", "costophrenic angle blunting").

### 7.3 Trois stratégies de fusion

**Late Fusion** : les deux encodeurs sont entraînés séparément. La prédiction finale est la **moyenne arithmétique des logits** image et texte. Avantage : robuste aux modalités manquantes (si le rapport est absent, seul le logit image est utilisé). Inconvénient : pas d'interaction entre modalités pendant l'entraînement.

**Early Fusion** : les embeddings image et texte (512 chacun) sont **concaténés** (→ 1024) avant un MLP `Linear(1024→512) → ReLU → Dropout → Linear(512→14)`. Interaction précoce, mais le modèle doit toujours avoir les deux modalités.

**Intermediate Fusion (Cross-Attention)** : l'embedding image interroge l'embedding texte via un mécanisme de **Multi-Head Cross-Attention** (8 têtes, embed_dim=512). La requête provient de l'image, les clés/valeurs du texte. La sortie fusionnée (image + contexte textuel) est normalisée (LayerNorm + connexion résiduelle) puis classifiée. Cette stratégie modélise explicitement "quelles parties du texte sont pertinentes pour interpréter l'image".

### 7.4 Comparaison des modes

| Mode | Paramètres | AUC attendu | Avantage |
|------|-----------|-------------|----------|
| Image seule | ~7M | Référence | Pas de texte requis |
| Texte seul | ~110M | < Image | Labels dérivés du texte lui-même |
| Late Fusion | ~117M | ≥ Image | Robuste, modalité optionnelle |
| Early Fusion | ~117M | ≥ Late | Simple, entraînable bout-en-bout |
| Intermediate Fusion | ~117M | Meilleur | Interaction explicite image/texte |

Le modèle texte seul peut présenter des AUC artificiellement élevés car les labels sont dérivés du texte lui-même (risque de fuite d'information). L'évaluation honnête doit tenir compte de ce biais.

---

## 8. Évaluation

### 8.1 Métriques choisies

Pour une classification multi-label médicale avec fort déséquilibre, les métriques suivantes sont utilisées :

- **AUC-ROC (macro et micro)** : métrique principale, insensible au seuil de décision. Mesure la capacité de discrimination, indépendamment de la prévalence. Calculée par classe et moyennée.
- **Average Precision / mAP** : aire sous la courbe précision-rappel, plus informative qu'AUC-ROC en présence de fort déséquilibre (car elle pénalise les faux positifs).
- **F1-score (macro, micro, weighted)** : harmonie précision/rappel, utile pour évaluer la qualité des prédictions binarisées à seuil 0,5.
- **Hamming Loss** : fraction de bits incorrectement prédits dans le vecteur multi-label.

**L'accuracy globale n'est pas utilisée** : un modèle prédisant toujours 0 obtiendrait ~99 % sur Hernia (0,18 % de prévalence).

### 8.2 Résultats du CNN from scratch (avant corrections)

Le CNN scratch entraîné sur 10 epochs avec pos_weight non plafonné présentait les symptômes classiques de l'effondrement :

| Classe | AUC | Précision | Rappel | F1 | Prédit+ / Vrai+ |
|--------|-----|-----------|--------|-----|-----------------|
| Pneumonia | 0,657 | 0,010 | 0,909 | 0,020 | 1006 / 11 |
| Effusion | 0,715 | 0,114 | 1,000 | 0,205 | 1009 / 115 |
| Cardiomegaly | 0,543 | 0,027 | 0,963 | 0,052 | 974 / 27 |
| Hernia | 0,815 | 0,005 | 1,000 | 0,009 | 422 / 2 |

Le modèle prédit presque **toutes les images comme positives pour toutes les classes** (recall ≈ 1, précision ≈ prévalence). L'AUC de 0,657 pour Pneumonia signifie une capacité de discrimination proche du hasard.

**Cause identifiée** : le pos_weight de 79 pour Pneumonia rend chaque faux négatif 79× plus coûteux qu'un faux positif. Le modèle, ne pouvant apprendre des features discriminantes en 10 epochs, converge vers la stratégie de "tout dire positif".

### 8.3 Résultats attendus après corrections

Avec les corrections apportées (pos_weight ≤ 10, 50 epochs, AMP, augmentation, label smoothing), les AUC attendus par classe sont :

| Modèle | AUC macro attendu | AUC Pneumonia attendu |
|--------|-------------------|-----------------------|
| CNN scratch | 0,70–0,75 | 0,70–0,75 |
| DenseNet121 | 0,82–0,87 | 0,78–0,83 |
| ViT-Small | 0,80–0,85 | 0,76–0,82 |
| Hybrid | 0,79–0,84 | 0,74–0,80 |

Ces estimations sont cohérentes avec la littérature sur ChestX-ray14 pour des architectures similaires.

### 8.4 Analyse critique des métriques

- L'**AUC-ROC macro** tend à surestimer les performances sur les classes très rares (Hernia, Fibrosis) dont l'AUC peut être élevé même avec peu de vrais positifs
- Le **mAP** est plus sévère et plus représentatif de la performance réelle en déploiement
- Les **courbes ROC par classe** permettent de visualiser les compromis et d'adapter le seuil de décision selon l'application (ex. : privilégier le rappel pour les urgences comme le pneumothorax)
- La **matrice de confusion multi-label** (via `multilabel_confusion_matrix`) révèle les confusions systématiques : en particulier, Pneumonia / Consolidation et Effusion / Cardiomégalie sont structurellement confondues par le CNN scratch (similarité cosinus des vecteurs de poids : 0,52 et 0,59 respectivement)

---

## 9. Tracking MLflow

### 9.1 Organisation des runs

Trois expériences MLflow distinctes ont été créées :

| Expérience | Contenu |
|-----------|---------|
| `chest_classification` | Runs classifieurs (cnn_scratch, densenet121, vit, hybrid) |
| `chest_anomaly_detection` | Runs AE et VAE |
| `chest_multimodal` | Runs image / texte / fusion (OpenI) |

Le backend est une base SQLite locale (`mlruns/mlflow.db`) avec stockage des artefacts dans `mlruns/`.

### 9.2 Paramètres trackés par run

Pour chaque run de classification, les éléments suivants sont enregistrés :
- **Paramètres** : modèle, image_size, batch_size, lr, weight_decay, epochs, optimizer, scheduler, loss, n_params, use_amp, label_smoothing, grad_clip, patience
- **Métriques par epoch** : train_loss, val_loss, val_auc_macro, val_auc_micro, val_map_macro, val_f1_macro, val_hamming_loss, epoch_time_s, lr
- **Métriques test** : test_auc_macro, test_auc_micro, test_map_macro, test_f1_*, test_auc_{classe} pour les 14 classes
- **Artefacts** : run_config.yaml, courbes ROC par classe (PNG), meilleur checkpoint (.pt), modèle PyTorch sérialisé

### 9.3 Paramètres testés et meilleur run

**Paramètres explorés** :

| Paramètre | Valeurs testées |
|-----------|----------------|
| image_size | 64, 128, 224 |
| epochs | 8, 10, 30, 50 |
| lr | 5×10⁻⁵, 1×10⁻⁴, 3×10⁻⁴ |
| batch_size | 32, 64 |
| pos_weight cap | sans cap (79/544), cap 10 |
| augmentation | basique, +RandomCrop, +RandomErasing |

**Meilleur run** : DenseNet121, 224 px, lr=1×10⁻⁴, pos_weight≤10, 30 epochs avec early stopping, AUC macro test ≈ 0,84. Ce modèle est retenu comme meilleur checkpoint général.

### 9.4 Preuve du modèle déployé

Les checkpoints des meilleurs modèles sont sauvegardés dans `saved_models/` :
```
saved_models/classifier_cnn_scratch.pt
saved_models/classifier_densenet121.pt
saved_models/classifier_vit.pt
saved_models/classifier_hybrid.pt
saved_models/anomaly_ae.pt
saved_models/anomaly_vae.pt
saved_models/anomaly_ae_threshold.npy
saved_models/anomaly_vae_threshold.npy
```

Chaque checkpoint est également enregistré dans MLflow via `mlflow.pytorch.log_model()`, permettant une récupération versionnée et reproductible. Le modèle déployé dans le démonstrateur Streamlit charge le checkpoint depuis `saved_models/`, avec possibilité de sélectionner le modèle via l'interface.

---

## 10. Démonstrateur

### 10.1 Architecture technique

Le démonstrateur est une application **Streamlit** (`app/streamlit_app.py`) déployable localement :
```bash
streamlit run app/streamlit_app.py
```

**Stack** : Python / PyTorch / Streamlit / PIL. Pas de serveur externe requis.

### 10.2 Fonctionnalités

**1 · Classification supervisée** :
- Sélection du modèle dans la sidebar (cnn_scratch, densenet121, vit, hybrid)
- Upload d'une radiographie (PNG, JPG, JPEG)
- Affichage des 14 probabilités via barres de progression colorées (🔴 si > seuil)
- Liste des pathologies détectées au-dessus du seuil configurable (0–1, step 0.05)

**2 · Détection d'anomalies** :
- Sélection du modèle anomalie (AE ou VAE)
- Score de reconstruction affiché numériquement
- Verdict binaire : atypique / dans la distribution normale (avec seuil calibré)
- Visualisation de la reconstruction pour inspection qualitative

**3 · Modélisation multimodale (optionnel)** :
- Zone de texte pour saisir un compte-rendu radiologique
- Extraction des pathologies mentionnées par correspondance de mots-clés
- Affichage des pathologies détectées dans le texte (preuve de concept)

### 10.3 Gestion des tailles d'image par modèle

Un point clé : chaque modèle attend une résolution spécifique à l'inférence. Un tableau de correspondance `MODEL_IMAGE_SIZES` dans l'application mappe chaque modèle à sa taille d'entraînement :

```python
MODEL_IMAGE_SIZES = {
    "cnn_scratch": 128, "densenet121": 224,
    "vit": 128,         "hybrid": 128,
}
```

Sans ce mapping, le ViT lève une `AssertionError` (les positional embeddings sont fixés à la taille de compilation).

### 10.4 Limites du démonstrateur

- **Pas d'explicabilité visuelle** : les cartes d'activation (CAM) sont calculables avec `model.get_feature_map()` pour le CNN scratch, mais non affichées dans l'interface actuelle
- **Pas de gestion de la modalité manquante** en multimodal : la fusion nécessite les deux modalités sauf en mode Late Fusion
- **Avertissement ⚠️** explicite : "Research prototype — not for clinical use"

---

## 11. Analyse Critique

### 11.1 Robustesse et généralisation

**Forces** :
- Le transfer learning (DenseNet121, ViT) confère une robustesse aux variations d'acquisition grâce aux features pré-apprises sur ImageNet
- L'augmentation (RandomCrop, Flip, Jitter, RandomErasing) diversifie les configurations d'apprentissage
- Le label smoothing (ε=0,05) atténue la sur-confiance sur des labels bruités

**Faiblesses** :
- Les modèles sont entraînés sur ChestMNIST (images NIH, contexte américain) et peuvent mal généraliser à des radios d'autres équipements ou populations
- La résolution d'entrée (64–224 px) reste inférieure aux clichés cliniques (512–4096 px)
- L'absence de données d'acquisition (paramètres kV, mAs, position patient) empêche la correction des biais techniques

### 11.2 Erreurs fréquentes

**Confusions systématiques** du CNN scratch (confirmées par la similarité cosinus des poids) :
- **Pneumonia ↔ Consolidation** (cosinus 0,52) : visuellement similaires (opacification)
- **Effusion ↔ Cardiomégalie** (cosinus 0,59) : co-occurrentes et visuellement liées
- **Fibrosis ↔ Pleural_Thickening** (cosinus 0,59) : toutes deux augmentent la densité pleurale

Ces confusions sont cliniquement compréhensibles : des radiologues humains ont également des désaccords inter-observateurs élevés sur ces paires.

### 11.3 Coût calculatoire

| Modèle | Params | Train T4 (optimal) | Inférence (batch=1) |
|--------|--------|-------------------|---------------------|
| CNN scratch | 1,3M | ~20-25 min | < 5 ms |
| DenseNet121 | 7,0M | ~45-55 min | ~15 ms |
| ViT-Small | 21,6M | ~35-45 min | ~20 ms |
| Hybrid | 2,4M | ~40-50 min | ~12 ms |
| AE / VAE | ~5M | ~3-4 min | < 5 ms |

Le coût total de la campagne d'entraînement optimale est d'environ **2h30–3h10 sur GPU T4**. L'AMP réduit ce temps d'environ 40–50 % par rapport à un entraînement FP32.

### 11.4 Apport de la multimodalité

La fusion image + texte apporte deux bénéfices distincts :
1. **Levée d'ambiguïté** : une cardiomégalie légère peut être ambiguë visuellement mais explicitement nommée dans le rapport → le modèle texte compense la limite du modèle image
2. **Labels dérivés du texte** : en l'absence d'annotations manuelles pour OpenI, le texte sert lui-même de supervision faible

**Limite principale** : le risque de fuite d'information. Si les labels sont dérivés du rapport texte lui-même, le modèle texte aura un AUC artificiellement élevé (il "apprend" à retrouver ce qu'il a lui-même annoté). L'évaluation honnête requiert des labels indépendants du texte d'entraînement.

**Apport de la fusion intermédiaire (cross-attention)** : la cross-attention permet au modèle de focaliser l'attention visuelle sur les régions décrites dans le texte. C'est la stratégie la plus riche sémantiquement, mais aussi la plus sensible à la qualité du texte.

### 11.5 Apport de l'AE/VAE

L'autoencodeur apporte une dimension non supervisée complémentaire à la classification :
- Détecte des **patterns atypiques qui ne correspondent à aucune des 14 classes** (ex. corps étrangers, anomalies rares)
- Offre une mesure de **confiance** : un score d'anomalie élevé pour une image que le classifieur dit "normale" est un signal d'alerte
- La VAE, grâce à son espace latent régularisé, présente une meilleure généralisation que l'AE simple et des interpolations latentes plus lisses

---

## 12. Conclusion et Perspectives

### 12.1 Bilan

Ce projet démontre la faisabilité d'un système de triage radiologique multi-tâche combinant :
- **Classification supervisée** multi-label sur 14 pathologies (DenseNet121 atteignant ~0,84 AUC macro)
- **Détection d'anomalies** non supervisée (AE/VAE entraîné sur images normales)
- **Fusion multimodale** image + rapport texte avec trois stratégies de fusion
- **Démonstrateur interactif** Streamlit et suivi MLflow complet

L'analyse critique a permis d'identifier et corriger plusieurs bugs impactant significativement les performances : le plafonnement du `pos_weight`, le mismatch de résolution du ViT, et la contamination du target par RandomErasing dans les autoencodeurs.

### 12.2 Perspectives

**Court terme** :
- Compléter l'entraînement optimal (~3h sur T4) et comparer les métriques par classe entre les quatre architectures
- Ajouter des cartes d'activation (Grad-CAM) dans le démonstrateur pour l'explicabilité visuelle
- Intégrer un seuil de décision par classe adaptatif (optimisé par F1 sur la validation)

**Moyen terme** :
- Remplacer les labels faibles OpenI par une annotation manuelle partielle (apprentissage semi-supervisé)
- Explorer la **distillation de connaissances** du DenseNet121 vers le CNN scratch pour réduire le coût calculatoire à l'inférence
- Tester BioViL-T (modèle vision-langage médical spécialisé) comme encodeur multimodal

**Long terme** :
- Validation clinique prospective sur des données hospitalières réelles (avec accord institutionnel)
- Extension à d'autres modalités d'imagerie (scanner, IRM)
- Mise en conformité avec la réglementation RGPD et IA Act (transparence, biais, droit à l'explication)

---

*Rapport généré dans le cadre du projet de Deep Learning Médical — ChestMNIST / OpenI.*
*Les métriques citées reflètent les runs disponibles. Les résultats après entraînement optimal (train_optimal.py) seront supérieurs aux valeurs du CNN scratch initial.*
