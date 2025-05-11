"""
Pipeline d'Intelligence Artificielle pour la prédiction de toxicité des champignons extraterrestres.
Compétition OFIA 2025.

<<<<<<< HEAD
Auteur: Gallemard Dylan ( Zeusfail )
Date: 2025-03-29

requirements.txt : 
numpy==1.24.3
pandas==2.0.3
torch==2.0.1
torchvision==0.15.2
scikit-learn==1.3.0
matplotlib==3.7.2
pillow==10.0.0
tqdm==4.65.0
joblib==1.3.1

Olympiade Française IA 2025
│
├── main.py                      # Script principal contenant la pipeline d'IA
│
├── data\                        # Dossier contenant les données tabulaires
│   ├── X_train.csv              # Données d'entraînement
│   ├── y_train.csv              # Étiquettes d'entraînement (toxicité)
│   └── X_test.csv               # Données de test
│
├── images\                      # Dossier contenant les images de champignons
│   ├── train\                   # Images pour l'entraînement
│   │   └── champignon_{id}.png  # Format des noms d'images
│   └── test\                    # Images pour le test
│       └── champignon_{id}.png  # Format des noms d'images
│
└── output\                      # Dossier pour les résultats et outputs (créé par le script)
    ├── models\                  # Modèles entraînés sauvegardés
    │   ├── best_model.pt        # Meilleur modèle entraîné
    │   └── preprocessor.pkl     # Préprocesseur sauvegardé
    ├── logs\                    # Logs et visualisations
    │   └── enhanced_training_history.png  # Graphique d'historique d'entraînement
    └── submission.csv           # Fichier de résultats pour soumission
=======
Auteur: Dylan Gallemard ( Zeusfail )
Date: 2025-03-29
>>>>>>> bb50886 (Final)
"""

# Import des bibliothèques nécessaires
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import warnings

# Ignorer les warnings non critiques pour nettoyer la sortie console
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Constantes globales pour la configuration du projet
SEED = 42                      # Seed pour assurer la reproductibilité
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Utilisation GPU si disponible
BATCH_SIZE = 64                # Taille des lots pour l'entraînement
NUM_EPOCHS = 100               # Nombre maximal d'époques d'entraînement
VAL_SIZE = 0.2                 # Proportion des données pour la validation
IMAGE_SIZE = 256               # Taille des images (pixels)
OUTPUT_DIR = "output"          # Dossier pour les sorties
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")  # Dossier pour les modèles
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")      # Dossier pour les logs
TEST_SAMPLE_SIZE = 10          # Nombre d'échantillons pour le mode test

# Fonction pour fixer tous les seeds (reproductibilité)
def set_seed(seed):
    """Fixe tous les seeds pour la reproductibilité"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Création des répertoires nécessaires au projet
for directory in [OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

def check_data_availability(args):
    """Vérifie que tous les fichiers et dossiers nécessaires sont présents
    
    Cette fonction vérifie l'existence des fichiers de données et des dossiers d'images
    avant de lancer le pipeline pour éviter les erreurs en cours d'exécution.
    
    Args:
        args: Arguments de ligne de commande contenant les chemins des fichiers
        
    Returns:
        bool: True si tous les éléments requis sont présents, sinon lève une exception
    """
    # Liste pour collecter les éléments manquants
    missing_elements = []
    
    # Vérification des fichiers CSV
    required_files = [
        (args.train_data, "Données d'entraînement (X_train.csv)"),
        (args.train_labels, "Étiquettes d'entraînement (y_train.csv)"),
        (args.test_data, "Données de test (X_test.csv)")
    ]
    
    for file_path, description in required_files:
        if not os.path.isfile(file_path):
            missing_elements.append(f"{description} : {file_path}")
    
    # Vérification des dossiers d'images
    required_dirs = [
        (args.train_images, "Dossier d'images d'entraînement"),
        (args.test_images, "Dossier d'images de test")
    ]
    
    for dir_path, description in required_dirs:
        if not os.path.isdir(dir_path):
            missing_elements.append(f"{description} : {dir_path}")
        else:
            # Vérifier que le dossier contient des images
            image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                missing_elements.append(f"{description} ne contient aucune image")
    
    # Si des éléments sont manquants, afficher un message d'erreur détaillé
    if missing_elements:
        error_message = "Éléments manquants détectés:\n" + "\n".join(missing_elements)
        error_message += "\n\nVeuillez vérifier la structure des dossiers et les chemins spécifiés."
        raise FileNotFoundError(error_message)
    
    print("Tous les fichiers et dossiers nécessaires sont présents")
    return True


class ChampignonPreprocessor:
    """Préprocesseur pour les données tabulaires des champignons
    
    Cette classe gère tout le prétraitement des données tabulaires:
    - Imputation des valeurs manquantes
    - Encodage des variables catégorielles et binaires
    - Standardisation des variables numériques
    """
    
    def __init__(self):
        # Définition des types de colonnes selon leur nature
        self.categorical_cols = ['odeur', 'texture', 'type_sol', 'effet_au_toucher']
        self.binary_cols = ['presence_insecte', 'a_l_air_delicieux_selon_renard']
        self.numerical_cols = ['x', 'y', 'poids', 'porosite', 'ph_du_jus', 'ph_du_sol', 'temperature_du_sol']
        
        # Pour stocker les colonnes effectivement disponibles dans les données
        self.fitted_categorical_cols = []
        self.fitted_binary_cols = []
        self.fitted_numerical_cols = []
        
        # Transformateurs qui seront initialisés lors de l'appel à fit()
        self.num_imputer = None    # Pour imputer les valeurs numériques manquantes
        self.cat_imputer = None    # Pour imputer les valeurs catégorielles manquantes
        self.scaler = None         # Pour standardiser les données numériques
        self.ohe = None            # Pour encoder en one-hot les variables catégorielles
        self.bin_encoder = None    # Pour encoder les variables binaires
    
    def fit(self, df):
        """Ajuste les transformateurs sur les données d'entraînement
        
        Args:
            df: DataFrame pandas contenant les données d'entraînement
            
        Returns:
            self: Le préprocesseur ajusté
        """
        # Identifier les colonnes disponibles dans le jeu de données
        self.fitted_categorical_cols = [col for col in self.categorical_cols if col in df.columns]
        self.fitted_binary_cols = [col for col in self.binary_cols if col in df.columns]
        self.fitted_numerical_cols = [col for col in self.numerical_cols if col in df.columns]
        
        # Traitement des variables catégorielles
        if self.fitted_categorical_cols:
            # Imputer les valeurs manquantes par la valeur la plus fréquente
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            cat_data = self.cat_imputer.fit_transform(df[self.fitted_categorical_cols])
            # Encodage one-hot des variables catégorielles
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.ohe.fit(cat_data)
        
        # Traitement des variables binaires
        if self.fitted_binary_cols:
            # Imputer les valeurs manquantes par la valeur la plus fréquente
            self.bin_imputer = SimpleImputer(strategy='most_frequent')
            bin_data = self.bin_imputer.fit_transform(df[self.fitted_binary_cols])
            # Encodage ordinal (0/1) des variables binaires
            self.bin_encoder = OrdinalEncoder()
            self.bin_encoder.fit(bin_data)
        
        # Traitement des variables numériques
        if self.fitted_numerical_cols:
            # Imputer les valeurs manquantes par la moyenne
            self.num_imputer = SimpleImputer(strategy='mean')
            num_data = self.num_imputer.fit_transform(df[self.fitted_numerical_cols])
            # Standardisation (moyenne 0, écart-type 1)
            self.scaler = StandardScaler()
            self.scaler.fit(num_data)
        
        return self
    
    def transform(self, df):
        """Transforme les données en utilisant les transformateurs ajustés
        
        Args:
            df: DataFrame pandas à transformer
            
        Returns:
            array: Les données transformées sous forme de tableau numpy
        """
        
        # Vérification que toutes les colonnes nécessaires sont présentes
        missing_cols = []
        for col_list in [self.fitted_categorical_cols, self.fitted_binary_cols, self.fitted_numerical_cols]:
            for col in col_list:
                if col not in df.columns:
                    missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans les données: {missing_cols}")
        
        # Initialisation des tableaux pour chaque type de données
        cat_encoded = np.array([]).reshape(df.shape[0], 0)
        bin_encoded = np.array([]).reshape(df.shape[0], 0)
        num_scaled = np.array([]).reshape(df.shape[0], 0)
        
        # Transformation des variables catégorielles
        if self.fitted_categorical_cols and self.cat_imputer is not None:
            cat_data = self.cat_imputer.transform(df[self.fitted_categorical_cols])
            cat_encoded = self.ohe.transform(cat_data)
        
        # Transformation des variables binaires
        if self.fitted_binary_cols and self.bin_imputer is not None:
            bin_data = self.bin_imputer.transform(df[self.fitted_binary_cols])
            bin_encoded = self.bin_encoder.transform(bin_data)
        
        # Transformation des variables numériques
        if self.fitted_numerical_cols and self.num_imputer is not None:
            num_data = self.num_imputer.transform(df[self.fitted_numerical_cols])
            num_scaled = self.scaler.transform(num_data)
        
        # Concaténation de toutes les features transformées
        X_processed = np.hstack([num_scaled, cat_encoded, bin_encoded])
        
        return X_processed
    
    def fit_transform(self, df):
        """Combine fit et transform en une seule opération
        
        Args:
            df: DataFrame pandas à ajuster et transformer
            
        Returns:
            array: Les données transformées
        """
        return self.fit(df).transform(df)
    
    def save(self, path):
        """Sauvegarde le préprocesseur pour utilisation ultérieure
        
        Args:
            path: Chemin où sauvegarder le préprocesseur
        """
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path):
        """Charge un préprocesseur préalablement sauvegardé
        
        Args:
            path: Chemin vers le préprocesseur sauvegardé
            
        Returns:
            ChampignonPreprocessor: Le préprocesseur chargé
        """
        return joblib.load(path)


class ChampignonDataset(Dataset):
    """Dataset PyTorch pour les données multimodales de champignons (images + tabulaires)
    
    Permet de charger et combiner des données tabulaires et d'images pour l'entraînement
    et l'inférence du modèle multimodal.
    """
    
    def __init__(self, tabular_data, img_dir, labels=None, transform=None, is_test=False, preprocessor=None):
        """
        Args:
            tabular_data: DataFrame pandas avec les données tabulaires
            img_dir: Chemin vers le dossier des images
            labels: DataFrame ou array avec les étiquettes de toxicité (si disponibles)
            transform: Transformations à appliquer aux images
            is_test: Si True, mode test (pas de labels attendus)
            preprocessor: Préprocesseur pour transformer les données tabulaires
        """
        self.tabular_data = tabular_data
        self.img_dir = img_dir
        self.labels = labels
        self.is_test = is_test
        self.preprocessor = preprocessor
        self.ids = tabular_data['id'].values
        
        # Prétraitement des données tabulaires si un préprocesseur est fourni
        if self.preprocessor is not None:
            self.processed_data = self.preprocessor.transform(tabular_data.drop('id', axis=1))
        else:
            self.processed_data = None
        
        # Transformations par défaut pour les images si aucune n'est spécifiée
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        """Retourne le nombre d'exemples dans le dataset"""
        return len(self.tabular_data)
    
    def __getitem__(self, idx):
        """Récupère un exemple (image + données tabulaires) à l'index spécifié
        
        Args:
            idx: Index de l'exemple à récupérer
            
        Returns:
            dict: Dictionnaire contenant l'image, les données tabulaires, l'ID et l'étiquette (si disponible)
        """
        # Récupération de l'ID du champignon
        img_id = self.ids[idx]
        
        # Chargement de l'image correspondante
        img_path = os.path.join(self.img_dir, f"champignon_{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        
        # Application des transformations d'image
        if self.transform:
            image = self.transform(image)
        
        # Récupération des données tabulaires
        if self.processed_data is not None:
            # Utiliser les données prétraitées par le préprocesseur
            tabular = torch.FloatTensor(self.processed_data[idx])
        else:
            # Fallback: utiliser uniquement les colonnes numériques
            numeric_cols = ['x', 'y', 'poids', 'porosite', 'ph_du_jus', 'ph_du_sol', 'temperature_du_sol', 
                           'ratio_ph', 'poids_porosite', 'distance_origine']
            tabular = torch.FloatTensor(self.tabular_data.iloc[idx][numeric_cols].values)
        
        # Construction du dictionnaire de retour
        item = {'image': image, 'tabular': tabular, 'id': img_id}
        
        # Ajout de l'étiquette (label) si disponible et pas en mode test
        if self.labels is not None and not self.is_test:
            if isinstance(self.labels, pd.DataFrame):
                label = torch.FloatTensor([self.labels.iloc[idx]['est_toxique']])
            else:
                label = torch.FloatTensor([self.labels[idx]])
            item['label'] = label
        
        return item
    
class FocalLoss(nn.Module):
    """Implémentation de la Focal Loss pour gérer le déséquilibre de classes
    
    La Focal Loss réduit la contribution des exemples faciles à classifier
    et se concentre davantage sur les exemples difficiles.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Args:
            alpha: Facteur de pondération pour la classe positive
            gamma: Facteur d'atténuation des exemples faciles (plus gamma est grand,
                  plus les exemples faciles sont atténués)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # BCE sans réduction pour appliquer la formule focal
        
    def forward(self, inputs, targets):
        """Calcule la focal loss entre les prédictions et les cibles
        
        Args:
            inputs: Prédictions du modèle (logits)
            targets: Valeurs cibles (0 ou 1)
            
        Returns:
            tensor: Valeur moyenne de la focal loss
        """
        # Calcul de la BCE loss standard
        BCE_loss = self.bce(inputs, targets)
        # pt est la probabilité prédite de la classe correcte
        pt = torch.exp(-BCE_loss)  # empêche les NaN quand la probabilité est 0
        # Formule de la Focal Loss
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        # Retourne la moyenne sur le batch
        return torch.mean(F_loss)

class MultimodalToxicityModel(nn.Module):
    """Modèle multimodal pour la prédiction de toxicité des champignons
    
    Ce modèle combine deux branches distinctes:
    1. Une branche CNN pour traiter les images des champignons
    2. Une branche pour traiter les données tabulaires (mesures physico-chimiques)
    
    Les deux types de données sont ensuite fusionnés via un mécanisme d'attention
    pour produire une prédiction finale sur la toxicité.
    """
    
    def __init__(self, tabular_input_size):
        """Initialise le modèle multimodal
        
        Args:
            tabular_input_size: Dimension des données tabulaires après prétraitement
        """
        super(MultimodalToxicityModel, self).__init__()
        
        # 1. Branche CNN basée sur EfficientNet-B0 (plus efficace que ResNet-18)
        # Utilisation de poids pré-entraînés sur ImageNet pour le transfer learning
        self.cnn_branch = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = self.cnn_branch.classifier[1].in_features
        # Remplace la couche de classification par des couches adaptées à notre tâche
        self.cnn_branch.classifier = nn.Sequential(
            nn.Dropout(0.4),                  # Réduit l'overfitting
            nn.Linear(num_ftrs, 256),         # Réduit la dimension des features
            nn.ReLU(),                        # Activation non-linéaire
            nn.BatchNorm1d(256),              # Normalisation pour stabiliser l'entraînement
            nn.Dropout(0.5)                   # Dropout plus fort en fin de réseau
        )
        
        # 2. Branche tabulaire avec architecture profonde pour capturer des interactions complexes
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # 3. Mécanisme d'attention pour pondérer l'importance des différentes features
        # L'attention permet au modèle de se concentrer sur les aspects les plus pertinents
        self.attention = nn.Sequential(
            nn.Linear(256 + 128, 384),        # Combine les features des deux branches
            nn.Tanh(),                        # Activation bornée entre -1 et 1
            nn.Linear(384, 384),              # Seconde couche d'attention
            nn.Softmax(dim=1)                 # Normalise les poids d'attention
        )
        
        # 4. Couches de fusion pour combiner les informations des deux modalités
        self.fusion_layers = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Dropout(0.5),                  # Fort dropout pour éviter l'overfitting
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 1)                  # Sortie unique: probabilité de toxicité
        )
    
    def forward(self, image, tabular):
        """Propagation avant du modèle
        
        Args:
            image: Tensor d'images de champignons [batch_size, channels, height, width]
            tabular: Tensor de données tabulaires [batch_size, tabular_input_size]
            
        Returns:
            Tensor: Logits pour la prédiction de toxicité [batch_size, 1]
        """
        # Extraire les features des images via le réseau CNN
        img_features = self.cnn_branch(image)
        
        # Extraire les features des données tabulaires
        tab_features = self.tabular_branch(tabular)
        
        # Concaténer les features des deux modalités
        combined = torch.cat([img_features, tab_features], dim=1)
        
        # Appliquer le mécanisme d'attention pour pondérer les features
        attention_weights = self.attention(combined)
        weighted_features = combined * attention_weights
        
        # Prédire la toxicité via les couches de fusion
        output = self.fusion_layers(weighted_features)
        
        return output


def create_data_augmentation():
    """Crée des transformations d'augmentation pour améliorer la généralisation
    
    L'augmentation de données permet d'enrichir artificiellement le jeu d'entraînement
    en appliquant diverses transformations aux images.
    
    Returns:
        tuple: (train_transform, test_transform) - Transformations pour l'entraînement et l'évaluation
    """
    # Transformations d'entraînement avec forte augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),               # Redimensionnement standard
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),# Recadrage aléatoire
        transforms.RandomHorizontalFlip(),                         # Retournement horizontal
        transforms.RandomVerticalFlip(),                           # Retournement vertical
        transforms.RandomRotation(20),                             # Rotation aléatoire
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Variation de couleur
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation aléatoire
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),  # Flou gaussien occasionnel
        transforms.ToTensor(),                                     # Conversion en tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisation ImageNet
        transforms.RandomErasing(p=0.2)                            # Effacement aléatoire de zones
    ])
    
    # Transformations de test plus simples (pas d'augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def create_ratio_features(df):
    """Crée des features dérivées basées sur des ratios entre variables
    
    L'ingénierie de features permet de capturer des relations non-linéaires
    entre différentes variables et d'améliorer les performances du modèle.
    
    Args:
        df: DataFrame pandas contenant les données brutes
        
    Returns:
        DataFrame: DataFrame original enrichi avec les nouvelles features
    """
    features = pd.DataFrame(index=df.index)
    
    # Éviter les divisions par zéro en ajoutant une petite constante
    epsilon = 1e-10
    
    # Ratio pH jus / pH sol (indicateur de différence d'acidité)
    features['ratio_ph'] = df['ph_du_jus'] / (df['ph_du_sol'] + epsilon)
    
    # Poids par unité de porosité (densité effective)
    features['poids_porosite'] = df['poids'] / (df['porosite'] + epsilon)
    
    # Coordonnées normalisées (distance à l'origine)
    features['distance_origine'] = np.sqrt(df['x']**2 + df['y']**2)
    
    # Ajouter ces features au DataFrame original
    return pd.concat([df, features], axis=1)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, 
                scheduler=None, early_stopping_patience=5, early_stopping_delta=0.001):
    """Entraîne le modèle multimodal avec techniques avancées
    
    Utilise:
    - Mixed Precision Training pour accélérer l'entraînement sur GPU
    - Early Stopping pour éviter l'overfitting
    - Test-Time Augmentation pour des prédictions plus robustes
    - Suivi détaillé des métriques de performance
    
    Args:
        model: Le modèle à entraîner
        train_loader: DataLoader pour les données d'entraînement
        val_loader: DataLoader pour les données de validation
        criterion: Fonction de perte
        optimizer: Optimiseur (AdamW recommandé)
        num_epochs: Nombre maximal d'époques
        device: Appareil d'exécution (CPU/GPU)
        scheduler: Planificateur de taux d'apprentissage
        early_stopping_patience: Nombre d'époques sans amélioration avant arrêt
        early_stopping_delta: Seuil d'amélioration minimal pour le early stopping
        
    Returns:
        tuple: (model, history) - Le modèle entraîné et l'historique d'entraînement
    """
    # Dictionnaire pour stocker l'historique d'entraînement
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'lr': []}
    best_val_auc = 0.0
    early_stopping_counter = 0
    early_stop = False

    # Initialiser le scaler pour Mixed Precision (accélère l'entraînement sur GPU)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        print(f"\nÉpoque {epoch+1}/{num_epochs}")

        # Mode entraînement
        model.train()
        train_loss = 0.0
        
        # Suivi du taux d'apprentissage actuel
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"Taux d'apprentissage: {current_lr:.7f}")

        # Barre de progression pour visualiser l'avancement
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, batch in progress_bar:
            images = batch['image'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()  # Réinitialiser les gradients
            
            # Utiliser Mixed Precision pour accélérer l'entraînement
            with torch.cuda.amp.autocast():
                outputs = model(images, tabular)
                loss = criterion(outputs, labels)

            # Rétropropagation avec scale pour éviter les underflows en précision mixte
            scaler.scale(loss).backward()
            
            # Clipper les gradients pour éviter l'explosion des gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Mise à jour des poids avec gestion de la précision mixte
            scaler.step(optimizer)
            scaler.update()
            
            # Mise à jour du planificateur si c'est OneCycleLR (mise à jour par batch)
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # Accumulation de la perte
            train_loss += loss.item() * images.size(0)

            # Mise à jour de la barre de progression
            progress_bar.set_description(f"Batch {batch_idx+1}/{len(train_loader)}")
            progress_bar.set_postfix(loss=loss.item())

        # Calcul de la perte moyenne sur l'époque
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validation avec Test-Time Augmentation pour des prédictions plus robustes
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():  # Désactive le calcul des gradients pendant la validation
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device)
                tabular = batch['tabular'].to(device)
                labels = batch['label'].to(device)
                
                # Test-Time Augmentation: moyenne de 3 prédictions différentes
                tta_preds = []
                
                # 1. Prédiction sur l'image originale
                outputs = model(images, tabular)
                tta_preds.append(outputs)
                
                # 2. Prédiction avec retournement horizontal
                flipped_images = torch.flip(images, [3])  # flip en largeur
                outputs_flip = model(flipped_images, tabular)
                tta_preds.append(outputs_flip)
                
                # 3. Prédiction avec découpage central
                outputs_center = model(images[:,:,16:-16,16:-16], tabular)
                tta_preds.append(outputs_center)
                
                # Moyenne des 3 prédictions pour une meilleure robustesse
                outputs = torch.mean(torch.stack(tta_preds), dim=0)
                
                # Calcul de la perte sur la validation
                loss = criterion(outputs, labels)
                
                # Accumulation des résultats
                val_loss += loss.item() * images.size(0)
                all_preds.extend(outputs.sigmoid().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calcul des métriques de validation
        val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(np.array(all_labels), np.array(all_preds))

        # Enregistrement des métriques dans l'historique
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        # Mise à jour du planificateur de type ReduceLROnPlateau (mise à jour par époque)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        # Affichage des métriques de l'époque
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Logique d'early stopping: arrêt si pas d'amélioration significative
        if val_auc > best_val_auc + early_stopping_delta:
            # Amélioration significative: réinitialiser le compteur et sauvegarder
            best_val_auc = val_auc
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pt'))
            print(f"Meilleur modèle sauvegardé (AUC: {val_auc:.4f})")
        else:
            # Pas d'amélioration significative: incrémenter le compteur
            early_stopping_counter += 1
            print(f"EarlyStopping: {early_stopping_counter}/{early_stopping_patience}")
            
            # Arrêt si le compteur atteint la patience définie
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping activé! Aucune amélioration depuis {early_stopping_patience} époques.")
                early_stop = True
                break

    # Charger le meilleur modèle si arrêt prématuré ou fin normale
    if early_stop:
        print(f"Chargement du meilleur modèle (époque {epoch+1-early_stopping_patience})")
    
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model.pt')))

    # Retourner le modèle entraîné et l'historique d'entraînement
    return model, history

def evaluate_model(model, test_loader, device):
    """Évalue le modèle sur les données de test
    
    Cette fonction effectue l'inférence sur un jeu de test fourni et
    produit un DataFrame contenant les IDs et les probabilités prédites.
    
    Args:
        model: Le modèle entraîné à évaluer
        test_loader: DataLoader contenant les données de test
        device: Dispositif d'exécution (CPU/GPU)
        
    Returns:
        DataFrame: Table contenant les IDs et les probabilités prédites
    """
    # Passer le modèle en mode évaluation (désactive dropout, etc.)
    model.eval()
    
    # Listes pour stocker les prédictions et les identifiants
    all_preds = []
    all_ids = []
    
    # Désactiver le calcul des gradients pour l'inférence (économise mémoire et calcul)
    with torch.no_grad():
        # Parcourir les lots de données de test avec une barre de progression
        for batch in tqdm(test_loader, desc="Évaluation"):
            # Transférer les données sur le dispositif approprié (CPU/GPU)
            images = batch['image'].to(device)
            tabular = batch['tabular'].to(device)
            ids = batch['id'].numpy()
            
            # Effectuer la prédiction et appliquer sigmoid pour obtenir une probabilité
            outputs = torch.sigmoid(model(images, tabular))
            
            # Collecter les résultats
            all_preds.extend(outputs.cpu().numpy())
            all_ids.extend(ids)
    
    # Créer le DataFrame final au format de soumission requis
    predictions = pd.DataFrame({
        'id': all_ids,
        'probabilite_toxique': np.concatenate(all_preds).flatten()
    })
    
    return predictions


def plot_training_history(history):
    """Trace les courbes d'entraînement avec plus d'informations
    
    Génère un graphique à 4 panneaux montrant:
    - Évolution de la fonction de perte (train vs validation)
    - Évolution de l'AUC ROC
    - Évolution du taux d'apprentissage
    
    Args:
        history: Dictionnaire contenant les métriques d'entraînement
    """
    # Créer une figure avec une taille appropriée
    plt.figure(figsize=(15, 8))
    
    # Panneau 1: Courbe de perte (train vs validation)
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Évolution de la perte')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    # Panneau 2: Courbe d'AUC ROC (avec ligne de référence aléatoire à 0.5)
    plt.subplot(2, 2, 2)
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3, label='Random')  # Ligne de référence à 0.5 (aléatoire)
    plt.title('Évolution du ROC AUC')
    plt.xlabel('Époque')
    plt.ylabel('AUC')
    plt.legend()
    
    # Panneau 3: Taux d'apprentissage (échelle logarithmique)
    plt.subplot(2, 2, 3)
    plt.plot(history['lr'], label='Learning Rate')
    plt.title('Évolution du taux d\'apprentissage')
    plt.xlabel('Époque')
    plt.ylabel('Taux d\'apprentissage')
    plt.yscale('log')  # Échelle logarithmique pour mieux visualiser les variations
    plt.legend()
    
    # Ajuster la mise en page et sauvegarder le graphique
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, 'enhanced_training_history.png'))
    plt.close()  # Fermer la figure pour libérer la mémoire

def main(args):
    """Fonction principale du pipeline
    
    Coordonne toutes les étapes du processus:
    - Chargement et prétraitement des données
    - Création et entraînement du modèle
    - Évaluation et génération des prédictions
    
    Args:
        args: Arguments de ligne de commande
        
    Returns:
        DataFrame: Prédictions finales
    """
    # Afficher les informations initiales
    print(f"Exécution sur: {DEVICE}")

    # Vérifier la disponibilité des données nécessaires
    check_data_availability(args)
    
    # Fixer les seeds pour la reproductibilité
    set_seed(SEED)
    print(f"Seed fixé à {SEED} pour reproductibilité")
    
    # Chronométrage de l'exécution
    start_time = time.time()
    
    #----- CHARGEMENT DES DONNÉES -----#
    print("\nChargement des données...")
    X_train = pd.read_csv(args.train_data)
    y_train = pd.read_csv(args.train_labels)
    X_test = pd.read_csv(args.test_data)
    
    # En mode test, utiliser seulement un petit échantillon pour accélérer le développement
    if args.test_mode:
        print(f"Mode test activé: utilisation de {TEST_SAMPLE_SIZE} échantillons")
        # Sélectionner un échantillon aléatoire avec seed fixe pour cohérence
        np.random.seed(SEED)
        test_indices = np.random.choice(len(X_test), TEST_SAMPLE_SIZE, replace=False)
        X_test = X_test.iloc[test_indices].reset_index(drop=True)
    
    #----- INGÉNIERIE DE FEATURES -----#
    print("Création de features dérivées...")
    # Créer des ratios et transformations non-linéaires pour améliorer le pouvoir prédictif
    X_train = create_ratio_features(X_train)
    X_test = create_ratio_features(X_test)
    
    #----- PRÉTRAITEMENT DES DONNÉES -----#
    print("Prétraitement des données tabulaires...")
    # Initialiser et ajuster le préprocesseur sur les données d'entraînement
    preprocessor = ChampignonPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train.drop('id', axis=1))
    X_test_processed = preprocessor.transform(X_test.drop('id', axis=1))
    
    # Sauvegarder le préprocesseur pour utilisation future (inférence)
    preprocessor.save(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    
    #----- SÉPARATION TRAIN/VALIDATION -----#
    # Diviser stratifiquement pour maintenir la distribution des classes
    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
        X_train, y_train, 
        test_size=VAL_SIZE,  # 20% pour la validation
        random_state=SEED,   # Pour reproductibilité
        stratify=y_train['est_toxique']  # Maintient proportion de champignons toxiques
    )
    
    # Prétraiter aussi les données de validation
    X_val_processed = preprocessor.transform(X_val_df.drop('id', axis=1))
    
    #----- PRÉPARATION DES DATASETS -----#
    # Créer les transformations d'augmentation pour les images
    train_transform, test_transform = create_data_augmentation()
    
    # Dataset d'entraînement avec augmentation de données
    train_dataset = ChampignonDataset(
        X_train_df, 
        args.train_images, 
        y_train_df, 
        transform=train_transform,  # Utilise forte augmentation pour l'entraînement
        preprocessor=preprocessor
    )
    
    # Dataset de validation (sans augmentation)
    val_dataset = ChampignonDataset(
        X_val_df, 
        args.train_images, 
        y_val_df, 
        transform=test_transform,  # Pas d'augmentation pour évaluation pure
        preprocessor=preprocessor
    )
    
    # Dataset de test (sans étiquettes)
    test_dataset = ChampignonDataset(
        X_test, 
        args.test_images, 
        transform=test_transform, 
        is_test=True,  # Indique qu'il s'agit de données de test sans étiquettes
        preprocessor=preprocessor
    )
    
    #----- CRÉATION DES DATALOADERS -----#
    # DataLoader pour l'entraînement avec mélange des données
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    # DataLoaders pour validation et test (sans mélange pour cohérence)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
    
    # Déterminer la taille des données tabulaires après prétraitement
    tabular_size = X_train_processed.shape[1]
    
    #----- CRÉATION ET ENTRAÎNEMENT DU MODÈLE -----#
    print("\nCréation du modèle multimodal...")
    # Initialiser le modèle multimodal avec la taille correcte d'entrée tabulaire
    model = MultimodalToxicityModel(tabular_input_size=tabular_size).to(DEVICE)
    
    # Focal Loss pour gérer le déséquilibre de classes
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    # AdamW pour une meilleure régularisation par rapport à Adam standard
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Calculer le nombre total d'étapes pour le planificateur d'apprentissage
    total_steps = NUM_EPOCHS * len(train_loader)

    # OneCycleLR pour un apprentissage efficace avec taux variable
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,         # Taux d'apprentissage maximal
        total_steps=total_steps,
        pct_start=0.3,        # 30% du temps pour atteindre le taux maximal
        anneal_strategy='cos'  # Annulation en cosinus pour transition douce
    )
    
    # Lancer l'entraînement du modèle
    print("\nDébut de l'entraînement...")
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        NUM_EPOCHS, 
        DEVICE,
        scheduler,
        early_stopping_patience=7  # Arrêt si pas d'amélioration après 7 époques
    )
    
    # Visualiser l'historique d'entraînement
    plot_training_history(history)
    
    #----- GÉNÉRATION DES PRÉDICTIONS FINALES -----#
    print("\nGénération des prédictions finales...")
    predictions = evaluate_model(trained_model, test_loader, DEVICE)
    
    # Sauvegarder les prédictions dans un fichier CSV pour soumission
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    predictions.to_csv(submission_path, index=False)
    print(f"Prédictions sauvegardées dans {submission_path}")
    
    # Afficher le temps total d'exécution
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\nTemps d'exécution total: {int(minutes)}m {int(seconds)}s")
    
    # Afficher un aperçu des prédictions générées
    print("\nAperçu des prédictions:")
    print(predictions.head(10))
    
    return predictions


if __name__ == "__main__":
    # Création du parseur d'arguments pour configurer l'exécution via ligne de commande
    parser = argparse.ArgumentParser(description="Pipeline de prédiction de toxicité des champignons")
    
    # Définition des arguments avec valeurs par défaut
    parser.add_argument("--train_data", type=str, default="data/X_train.csv", help="Chemin vers X_train.csv")
    parser.add_argument("--train_labels", type=str, default="data/y_train.csv", help="Chemin vers y_train.csv")
    parser.add_argument("--test_data", type=str, default="data/X_test.csv", help="Chemin vers X_test.csv")
    parser.add_argument("--train_images", type=str, default="images/train", help="Dossier des images d'entraînement")
    parser.add_argument("--test_images", type=str, default="images/test", help="Dossier des images de test")
    parser.add_argument("--test_mode", action="store_true", help="Mode test sur 10 échantillons")
    
    # Analyser les arguments et exécuter la fonction principale
    args = parser.parse_args()
    main(args)