import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import register_keras_serializable
from ASR.hparams import *


# String lookup layers
char_to_num = keras.layers.StringLookup(vocabulary=list(characters), oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def Fpreprocess_audio(wav_file):
    """
    Preprocesses a single audio file for ASR.
    
    Parameters:
    - wav_file (str): Path to the audio file (WAV format).
    
    Returns:
    - spectrogram (Tensor): Preprocessed spectrogram.
    """
    file = tf.io.read_file(wav_file)
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    
    if tf.shape(audio)[0] < fft_length:
        pad_amount = fft_length - tf.shape(audio)[0]
        audio = tf.pad(audio, paddings=[[0, pad_amount]])
    
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    
    return spectrogram

def Apreprocess_audio(audio_array):
    """
    Preprocesses a NumPy array of audio samples for ASR.
    
    Parameters:
    - audio_array (np.ndarray): Array of audio samples.
    
    Returns:
    - spectrogram (Tensor): Preprocessed spectrogram.
    """
    # Convert the NumPy array to a TensorFlow tensor
    audio = tf.convert_to_tensor(audio_array, dtype=tf.float32)
    
    if tf.shape(audio)[0] < fft_length:
        pad_amount = fft_length - tf.shape(audio)[0]
        audio = tf.pad(audio, paddings=[[0, pad_amount]])
    
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    
    return spectrogram

def decode_batch_predictions(pred):
    """
    Decodes batch predictions into text using character lookup.
    
    Parameters:
    - pred (Tensor): Predicted output from the model.
    
    Returns:
    - output_text (list): List of decoded text predictions.
    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

@register_keras_serializable()
def CTCLoss(y_true, y_pred):
    """
    Custom CTC Loss function for sequence prediction.
    
    Parameters:
    - y_true (Tensor): True labels.
    - y_pred (Tensor): Predicted outputs from the model.
    
    Returns:
    - loss (Tensor): Calculated CTC loss.
    """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss