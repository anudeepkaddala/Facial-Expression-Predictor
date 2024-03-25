import numpy as np
import os
import cv2
import uuid
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

# Flask Integration
app = Flask(__name__)

# Load the trained model
model = load_model("my_emotion_model_updated.h5")

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(image)
    resized_image = cv2.resize(equalized_image, (48, 48))
    normalized_image = resized_image / 255.0
    reshaped_image = np.reshape(normalized_image, (1, 48, 48, 1))
    return reshaped_image

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the uploaded file
            file = request.files['file']
            
            # Save the file to a unique temporary location
            file_path = 'temp_{}.jpg'.format(uuid.uuid4())
            file.save(file_path)

            # Preprocess the image
            processed_image = preprocess_image(file_path)

            # Debug: Print the processed image shape
            print("Processed Image Shape:", processed_image.shape)

            # Make a prediction
            prediction = model.predict(processed_image)
            predicted_class = emotions[np.argmax(prediction)]

            # Debug: Print the prediction
            print("Raw Prediction:", prediction)
            print("Predicted Class:", predicted_class)

            # Remove the temporary file
            os.remove(file_path)

            # Return the predicted class
            return jsonify({'emotion': predicted_class})

        except Exception as e:
            # Debug: Print any exceptions for debugging
            print("Error:", str(e))
            return jsonify({'error': 'Something went wrong'})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
