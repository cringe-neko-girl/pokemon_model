import os
import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image

# Load the pre-trained model
model = load_model('pokemon_model.keras')  # Make sure to load the model you saved

# Get class names from folder names in t_data/test
class_names = [folder for folder in os.listdir("t_data/test") if os.path.isdir(os.path.join("t_data/test", folder))]

# Infinite loop to accept user input
while True:
    # Get the image URL from the user
    image_url = input("Enter the image URL (or 'exit' to quit): ")
    
    if image_url.lower() == 'exit':
        print("Exiting the program.")
        break  # Exit the loop if the user types 'exit'
    
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Check if the request was successful

        # Open the image with PIL
        img = Image.open(BytesIO(response.content))

        # Resize the image to match the input size of the model (150x150)
        img = img.resize((150, 150))

        # Convert the image to a NumPy array and normalize it
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]  # Get the predicted class

        print(f"Predicted class: {predicted_class}")
    
    except Exception as e:
        print(f"Error loading or processing the image: {e}")
