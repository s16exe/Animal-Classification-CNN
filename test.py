import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model("friday.h5")

# Class names (Replace this with your actual class names in order)

data = tf.keras.utils.image_dataset_from_directory(
    "animals",  # Path to dataset
    image_size=(128, 128),
    batch_size=32
)

class_names = data.class_names
print(class_names)  # Verify the list of class names

# Function to preprocess a single image
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)  # Load image and resize
    img_array = img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class of an image
def predict_image(image_path):
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions, axis=1)  # Get class index
    class_name = class_names[predicted_class[0]]  # Map index to class name
    confidence = np.max(predictions)  # Confidence score
    return class_name, confidence

# Test with an example image
image_path = "C:\\Users\\subra\\clge\\wood2.jpg"  # Path to the image to classify
predicted_class, confidence = predict_image(image_path)

print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
