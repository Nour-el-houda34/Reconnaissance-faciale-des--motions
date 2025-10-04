from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Charger le modèle pré-entraîné
model = load_model("expression_recognition_model.h5")

# Catégories d'expressions
categories = ['angry', 'happy', 'sad', 'surprise', 'neutral', 'fear']

# Configurer les dossiers d'upload et de résultats
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Modèle Haar Cascade pour détecter les visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Erreur : Impossible de charger l'image."

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Vérifier si l'image est déjà au format (48x48) comme FER2013
        if gray.shape == (48, 48):
            face = gray / 255.0  # Normaliser
            face = np.expand_dims(face, axis=-1)  # Ajouter une dimension pour les niveaux de gris
            face = np.expand_dims(face, axis=0)   # Ajouter une dimension pour le batch

            # Prédire l'expression
            prediction = model.predict(face)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_expression = categories[predicted_class[0]]

            # Générer une image avec annotation
            result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
            annotated_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convertir en RGB pour affichage
            cv2.putText(annotated_image, predicted_expression, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imwrite(result_image_path, annotated_image)

            return result_image_path, [predicted_expression]

        # Si ce n'est pas une image de 48x48, appliquer Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)
        if len(faces) == 0:
            return None, "Aucun visage détecté dans l'image."

        detected_expressions = []

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48)) / 255.0  # Redimensionner et normaliser
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            # Prédire l'expression
            prediction = model.predict(face)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_expression = categories[predicted_class[0]]
            detected_expressions.append(predicted_expression)

            # Annoter l'image
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, predicted_expression, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Sauvegarder l'image annotée
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
        cv2.imwrite(result_image_path, image)

        return result_image_path, detected_expressions

    except Exception as e:
        return None, f"Erreur lors du traitement de l'image : {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    result_image_path, detected_expressions = process_image(file_path)

    if result_image_path is None:
        return jsonify({'error': detected_expressions}), 400

    # Générer un lien pour télécharger l'image annotée
    image_url = url_for('get_result_image', filename='result.jpg', _external=True)
    return jsonify({'expressions': detected_expressions, 'image': image_url})


@app.route('/results/<filename>')
def get_result_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
