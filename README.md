# 🎭 Reconnaissance Faciale des Émotions

Ce projet implémente un **système de reconnaissance automatique des expressions faciales** à partir d’images et de flux vidéo en temps réel, en utilisant des **Réseaux de Neurones Convolutionnels (CNN)** et une **application web Flask**.

Il a été réalisé dans un **cadre académique** (Master ISI – Data Mining).

---

## 📌 Objectif du Projet

L’objectif principal est de :
- détecter des visages humains dans des images ou une vidéo
- classifier automatiquement l’émotion faciale associée
- fournir une **interface web simple** permettant :
  - l’analyse d’images statiques
  - la reconnaissance des émotions en temps réel via webcam

Les émotions reconnues sont :
**Angry, Happy, Sad, Surprise, Neutral, Fear**

---

## 🧠 Principe de Fonctionnement

1. Chargement du dataset FER-2013
2. Prétraitement des images (48×48, niveaux de gris, normalisation)
3. Entraînement d’un modèle CNN
4. Sauvegarde du modèle entraîné
5. Déploiement via une application Flask
6. Prédiction des émotions sur images ou flux vidéo

---

## 📂 Contenu du Projet
```text

Reconnaissance-faciale-des-motions/
│
├── expression_recognition_model.h5
│   # Modèle entraîné de reconnaissance des expressions faciales (Keras / TensorFlow)
│
├── jupyter.ipynb
│   # Notebook Jupyter utilisé pour l’entraînement, les tests ou l’analyse des données
│
├── live-expression-recognition/
│   │
│   ├── app.py
│   │   # Application principale (Flask) pour lancer l’interface web
│   │
│   ├── live.py
│   │   # Script de reconnaissance des expressions faciales en temps réel via la webcam
│   │
│   ├── convert_model.py
│   │   # Script de conversion ou d’adaptation du modèle (ex : format, compatibilité)
│   │
│   ├── uploads/
│   │   # Dossier contenant les images importées par l’utilisateur
│   │
│   ├── results/
│   │   # Résultats générés après la reconnaissance des expressions
│   │
│   └── templates/
│       ├── index.html
│       │   # Page d’accueil de l’application web
│       │
│       └── live.html
│           # Interface de reconnaissance faciale en temps réel
│
├── .idea/
│   # Fichiers de configuration de l’IDE 
│
└── README.md
  ```
 

---
## 🛠️ Technologies Utilisées

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Flask

---

## ▶️ Installation et Exécution (Sur votre machine)

### 🔹 1. Cloner le dépôt
```bash
git clone https://github.com/Nour-el-houda34/Reconnaissance-faciale-des--motions.git
cd Reconnaissance-faciale-des--motions
```
###🔹 2. Créer un environnement virtuel
```bash
python -m venv venv
```
## ▶️ Installation et Exécution

### 🔹 2. Créer un environnement virtuel
```bash
    python -m venv venv
```
#### Sous Windows
```bash
    venv\Scripts\activate
```
#### Sous Linux / macOS
```bash
    source venv/bin/activate
```
---

### 🔹 3. Installer les dépendances
```bash
    pip install tensorflow opencv-python flask numpy pandas scikit-learn matplotlib

```
---

### 🔹 4. Lancer l’application web
```bash
    python app.py
```
Ouvrir le navigateur à l’adresse :

 ```bash
   http://localhost:5000
```
---

## 🌐 Fonctionnalités de l’Application

-  Upload d’images pour analyse  
-  Détection automatique des visages  
-  Affichage de l’émotion prédite  
-  Reconnaissance faciale en temps réel via webcam  

---

## 📚 Contexte Académique

Projet réalisé dans le cadre du **Master ISI – Machine Learning**
