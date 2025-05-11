=====================================================
PRÉDICTION DE TOXICITÉ DES CHAMPIGNONS EXTRATERRESTRES
Olympiade Française IA 2025
=====================================================

Auteur: Zeusfail // Gallemard Dylan
Date: 28 Mars 2025

DESCRIPTION
-----------
Ce projet implémente un modèle d'IA multimodal pour prédire la toxicité des champignons extraterrestres à partir de:
- Images de champignons
- Données tabulaires (mesures physico-chimiques)

Le modèle combine un réseau CNN EfficientNet-B0 pour les images et un réseau profond pour les données tabulaires,
fusionnés par un mécanisme d'attention pour produire des prédictions robustes.

STRUCTURE DES DOSSIERS
----------------------
- data/
  - X_train.csv (données tabulaires d'entraînement)
  - y_train.csv (étiquettes d'entraînement)
  - X_test.csv (données tabulaires de test)
- images/
  - train/ (images d'entraînement: champignon_<id>.png)
  - test/ (images de test: champignon_<id>.png)
- output/ (créé automatiquement)
  - models/ (modèles sauvegardés)
  - logs/ (visualisations et logs)

INSTALLATION
------------
1. Créez un environnement virtuel Python 3.10.0:
   python -m venv venv
   
2. Activez l'environnement:
   Windows: venv\Scripts\activate
   Linux/Mac: source venv/bin/activate
   
3. Installez les dépendances:
   pip install -r requirements.txt

UTILISATION
-----------
Exécution standard:
   python main.py

Options disponibles:
   --train_data     Chemin vers X_train.csv (défaut: data/X_train.csv)
   --train_labels   Chemin vers y_train.csv (défaut: data/y_train.csv)
   --test_data      Chemin vers X_test.csv (défaut: data/X_test.csv)
   --train_images   Dossier des images d'entraînement (défaut: images/train)
   --test_images    Dossier des images de test (défaut: images/test)
   --test_mode      Activer le mode test rapide (10 échantillons)

Exemple avec chemins personnalisés:
   python main.py --train_data donnees/X_train.csv --train_images mes_images/entrainement

SORTIES
-------
- output/submission.csv: Fichier de prédictions au format demandé pour la compétition
- output/models/best_model.pt: Meilleur modèle sauvegardé
- output/models/preprocessor.pkl: Préprocesseur sauvegardé
- output/logs/enhanced_training_history.png: Visualisation des métriques d'entraînement

RESSOURCES REQUISES
------------------
- GPU recommandé pour l'entraînement (CUDA compatible)
- RAM: minimum 16 GB
- Espace disque: minimum 10 GB
- Temps d'entraînement: ~1-2 heures selon le matériel

L'entraînement utilise Mixed Precision pour optimiser l'utilisation GPU.
Activez le mode test (--test_mode) pendant le développement pour des cycles plus rapides.