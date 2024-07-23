from tensorflow import keras
import tensorflow as tf
import json
from ASR.preprocessing import Apreprocess_audio, decode_batch_predictions, CTCLoss


def load_asr_model(config_path, weights_path):
    with open(config_path, 'r') as json_file:
        model_config = json.load(json_file)
    model = keras.models.model_from_json(json.dumps(model_config))
    model.load_weights(weights_path)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=CTCLoss)
    return model

# Function to predict text for an audio segment
def predict_text(segment_audio, model):
    # Preprocess the segment audio
    spectrogram = Apreprocess_audio(segment_audio)
    spectrogram = tf.expand_dims(spectrogram, axis=0)  # Add the batch dimension
    
    # Predict the text
    prediction = model.predict(spectrogram, verbose=0)
    decoded_text = decode_batch_predictions(prediction)
    
    return decoded_text[0]