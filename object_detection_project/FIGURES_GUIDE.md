# Guide des Figures et Tableaux pour le Rapport

Ce document liste toutes les figures, tableaux et captures d'écran nécessaires pour compléter le rapport LaTeX.

## Section 2 : Configuration GPU

### Figure 2.1 : Vérification CUDA
- **Type** : Capture d'écran
- **Contenu** : Sortie de la commande `nvcc --version`
- **Description** : Doit montrer CUDA 12.6 installé

### Figure 2.2 : Vérification PyTorch CUDA
- **Type** : Capture d'écran
- **Contenu** : Sortie du code de vérification GPU
- **Description** : Doit montrer :
  - CUDA disponible: True
  - Nom du GPU: NVIDIA GeForce RTX 3060 Laptop GPU
  - Version CUDA: 12.6
  - Mémoire GPU: 6.44 GB

## Section 3 : Collecte des Données

### Tableau 3.1 : Statistiques du Dataset
- **Type** : Tableau
- **Colonnes** : Split | Images | Labels | Pourcentage
- **Lignes** : Train, Valid, Test, Total
- **Source** : Sortie du notebook 01_data_collection.ipynb

### Tableau 3.2 : Distribution des Classes
- **Type** : Tableau
- **Colonnes** : Classe | Nombre d'instances | Pourcentage
- **Lignes** : head, helmet, person
- **Source** : Sortie du notebook 01_data_collection.ipynb

## Section 4 : EDA

### Figure 4.1 : Distribution des Images par Split
- **Type** : Graphique en barres
- **Source** : Notebook 02_eda.ipynb, Cell 1
- **Description** : Barres pour train, valid, test

### Figure 4.2 : Distribution des Classes
- **Type** : Graphique en barres
- **Source** : Notebook 02_eda.ipynb, Cell 4
- **Description** : Nombre d'instances par classe (head, helmet, person)

### Figure 4.3 : Distribution des Classes par Split
- **Type** : Graphique en barres groupées
- **Source** : Notebook 02_eda.ipynb, Cell 4
- **Description** : Distribution de chaque classe dans train/valid/test

### Figure 4.4 : Histogramme des Largeurs d'Images
- **Type** : Histogramme
- **Source** : Notebook 02_eda.ipynb, Cell 3
- **Description** : Distribution des largeurs en pixels

### Figure 4.5 : Histogramme des Hauteurs d'Images
- **Type** : Histogramme
- **Source** : Notebook 02_eda.ipynb, Cell 3
- **Description** : Distribution des hauteurs en pixels

### Figure 4.6 : Distribution des Ratios d'Aspect
- **Type** : Histogramme
- **Source** : Notebook 02_eda.ipynb, Cell 3
- **Description** : Distribution des ratios largeur/hauteur

### Tableau 4.1 : Top 10 Résolutions d'Images
- **Type** : Tableau
- **Colonnes** : Largeur | Hauteur | Nombre d'occurrences
- **Source** : Notebook 02_eda.ipynb, Cell 3

### Figure 4.7 : Histogramme Largeur des Boîtes (normalisé)
- **Type** : Histogramme
- **Source** : Notebook 02_eda.ipynb, Cell 4
- **Description** : Distribution des largeurs normalisées (0-1)

### Figure 4.8 : Histogramme Hauteur des Boîtes (normalisé)
- **Type** : Histogramme
- **Source** : Notebook 02_eda.ipynb, Cell 4
- **Description** : Distribution des hauteurs normalisées (0-1)

### Figure 4.9 : Histogramme Aire des Boîtes (normalisé)
- **Type** : Histogramme
- **Source** : Notebook 02_eda.ipynb, Cell 4
- **Description** : Distribution des aires normalisées

### Figure 4.10 : Ratio d'Aspect par Classe
- **Type** : 3 histogrammes (un par classe)
- **Source** : Notebook 02_eda.ipynb, Cell 4
- **Description** : head, helmet, person

### Tableau 4.2 : Catégorisation des Boîtes
- **Type** : Tableau croisé
- **Lignes** : head, helmet, person
- **Colonnes** : small, medium, large
- **Source** : Notebook 02_eda.ipynb, Cell 4

### Figure 4.11 : Exemples d'Images Annotées
- **Type** : Grille 2x2 ou 2x3
- **Source** : Notebook 02_eda.ipynb, Cell 5
- **Description** : 
  - 4-6 images avec boîtes de détection
  - Couleurs : Rouge (head), Vert (helmet), Cyan (person)
  - Une image par split (train/valid/test)

## Section 5 : Architecture YOLO

### Figure 5.1 : Schéma de l'Architecture YOLOv8
- **Type** : Diagramme
- **Description** : 
  - Backbone (CSPDarknet)
  - Neck (PANet)
  - Head (Detection)
  - Flux de données
- **Source** : Créer un diagramme ou utiliser une image de référence

### Figure 5.2 : Comparaison CPU vs GPU
- **Type** : Graphique en barres
- **Description** : 
  - Temps d'inference (ms) : CPU vs GPU
  - FPS : CPU vs GPU
- **Source** : Tests de performance

## Section 6 : Entraînement

### Figure 6.1 : Courbe de Perte Totale
- **Type** : Graphique linéaire
- **Source** : runs/detect/ppe_detection_phase2/results.png
- **Description** : Train loss et Val loss vs époques

### Figure 6.2 : Courbe mAP@0.5
- **Type** : Graphique linéaire
- **Source** : runs/detect/ppe_detection_phase2/results.png
- **Description** : mAP@0.5 vs époques

### Figure 6.3 : Courbe mAP@0.5:0.95
- **Type** : Graphique linéaire
- **Source** : runs/detect/ppe_detection_phase2/results.png
- **Description** : mAP@0.5:0.95 vs époques

### Figure 6.4 : Courbes Précision et Rappel
- **Type** : Graphique linéaire
- **Source** : runs/detect/ppe_detection_phase2/results.png
- **Description** : Precision et Recall vs époques

### Tableau 6.1 : Métriques Finales
- **Type** : Tableau
- **Colonnes** : Métrique | Valeur
- **Lignes** : 
  - mAP@0.5
  - mAP@0.5:0.95
  - Precision
  - Recall
  - F1-score
- **Source** : Validation finale du modèle

### Tableau 6.2 : Métriques par Classe
- **Type** : Tableau
- **Colonnes** : Classe | Precision | Recall | mAP@0.5 | mAP@0.5:0.95
- **Lignes** : head, helmet, person
- **Source** : Validation finale du modèle

## Section 7 : Évaluation

### Figure 7.1 : Matrice de Confusion Normalisée
- **Type** : Matrice de confusion (heatmap)
- **Source** : runs/detect/ppe_detection_phase2/confusion_matrix_normalized.png
- **Description** : Matrice normalisée avec pourcentages

### Figure 7.2 : Matrice de Confusion
- **Type** : Matrice de confusion (heatmap)
- **Source** : runs/detect/ppe_detection_phase2/confusion_matrix.png
- **Description** : Matrice avec nombres absolus

### Tableau 7.1 : Résultats sur l'Ensemble de Test
- **Type** : Tableau
- **Colonnes** : Métrique | Valeur
- **Lignes** :
  - Images testées
  - Total détections
  - Détections head
  - Détections helmet
  - Détections person
  - Taux moyen par image
- **Source** : Notebook 04_predict_ppe.ipynb, Cell 4

### Figure 7.3 : Exemples de Prédictions (6-8 images)
- **Type** : Grille 2x3 ou 2x4
- **Source** : runs/detect/ppe_test_predictions/
- **Description** : 
  - Mélange de cas réussis et échecs
  - Différentes conditions d'éclairage/angle
  - Boîtes colorées selon la classe

## Section 8 : Inference Temps Réel

### Figure 8.1 : Capture d'Écran Inference Vidéo
- **Type** : Capture d'écran
- **Description** : 
  - Fenêtre OpenCV avec détections en temps réel
  - FPS affiché
  - Nombre de détections affiché
- **Source** : Capture pendant l'exécution du notebook

### Figure 8.2 : Graphique FPS au Fil du Temps
- **Type** : Graphique linéaire
- **Description** : FPS vs temps (secondes)
- **Source** : Mesures pendant l'inference vidéo

### Tableau 8.1 : Comparaison CPU vs GPU
- **Type** : Tableau
- **Colonnes** : Configuration | FPS | Temps/Frame (ms) | Utilisation Mémoire
- **Lignes** :
  - CPU
  - GPU FP32
  - GPU FP16
- **Source** : Tests de performance

### Figure 8.3 : Comparaison FPS CPU vs GPU
- **Type** : Graphique en barres
- **Description** : FPS pour CPU, GPU FP32, GPU FP16
- **Source** : Tests de performance

### Figure 8.4 : Utilisation GPU pendant Inference
- **Type** : Graphique linéaire
- **Description** : 
  - Utilisation GPU (%) vs temps
  - Mémoire utilisée (GB) vs temps
- **Source** : NVIDIA-SMI ou monitoring GPU

## Section 9 : Analyse GPU

### Tableau 9.1 : Comparaison Détaillée CPU vs GPU
- **Type** : Tableau
- **Colonnes** : Opération | CPU (ms) | GPU (ms) | Accélération
- **Lignes** :
  - Preprocessing
  - Inference
  - Postprocessing
  - Total
- **Source** : Tests de performance

### Figure 9.1 : Impact de la Précision Mixte
- **Type** : Graphique en barres groupées
- **Description** : 
  - FPS : FP32 vs FP16
  - Mémoire : FP32 vs FP16
- **Source** : Tests de performance

### Figure 9.2 : Débit vs Taille de Batch
- **Type** : Graphique linéaire
- **Description** : Throughput (images/sec) vs Batch Size
- **Source** : Tests avec différentes tailles de batch

## Instructions pour l'Insertion

1. **Format des images** : PNG ou JPG, résolution minimale 300 DPI
2. **Nommage** : Utiliser des noms descriptifs (ex: `cuda_verification.png`)
3. **Taille** : Ajuster avec `\includegraphics[width=0.8\textwidth]{...}`
4. **Légendes** : Toutes les figures doivent avoir des légendes descriptives
5. **Références** : Utiliser `\ref{fig:...}` pour référencer les figures dans le texte

## Notes Importantes

- Toutes les figures doivent être de haute qualité
- Les graphiques doivent être lisibles (polices suffisamment grandes)
- Les couleurs doivent être distinctes et accessibles
- Les tableaux doivent être bien formatés et alignés
- Toutes les métriques doivent être cohérentes avec les résultats réels


