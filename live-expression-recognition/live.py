from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import io
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Charger le modèle pré-entraîné
model = load_model("expression_recognition_model.h5")
categories = ['angry', 'happy', 'sad', 'surprise', 'neutral', 'fear']

# Modèle Haar Cascade pour détecter les visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def process_frame(frame):
    # Décoder l'image base64
    decoded_data = base64.b64decode(frame.split(",")[1])
    np_data = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)

    if len(faces) == 0:
        return None, "Aucun visage détecté."

    detected_expressions = []
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48)) / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Prédire l'expression
        prediction = model.predict(face)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_expression = categories[predicted_class]
        detected_expressions.append(predicted_expression)

        # Annoter l'image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, predicted_expression, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    _, buffer = cv2.imencode('.jpg', image)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    return frame_encoded, detected_expressions[0]


@app.route('/')
def index():
    return render_template('live.html')


@socketio.on('frame')
def handle_frame(data):
    frame = data.get('frame')
    if not frame:
        return

    processed_frame, expression = process_frame(frame)
    if processed_frame:
        socketio.emit('emotion', {'emotion': expression, 'frame': f"data:image/jpeg;base64,{processed_frame}"})
    else:
        socketio.emit('emotion', {'emotion': 'Aucun visage détecté', 'frame': None})


if __name__ == '__main__':
    socketio.run(app, debug=True)
