import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the existing Keras model
model = load_model('/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/model_optimal2.h5')

# Define the directory to save the TensorFlow SavedModel
saved_model_path = '/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/saved_models'

# Export the Keras model as a TensorFlow SavedModel
model.export(saved_model_path)

# Convert the TensorFlow SavedModel to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()

# Save the converted model as a .tflite file
with open('/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/model.tflite', 'wb') as f:
    f.write(tflite_model)
