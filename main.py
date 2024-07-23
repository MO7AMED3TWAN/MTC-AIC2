from Diarization import Dpreprocess_wav, VoiceEncoder
from ASR.preprocessing import * 
from ASR.hparams import * 
from Diarization.hparams import sampling_rate
from Diarization.audio import *
from Diarization.voice_encoder import *

from pathlib import Path
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import librosa.display



#give the file path to your audio file and for the model
audio_file_path = './Data/DataForDirization/audio_sample_20.wav'
wav_path = Path(audio_file_path)

modelD_file_path = './Models/DIRIZATIONMODEL.pt'
modelD_file_path= Path(modelD_file_path)

modelA_file_path = './Models/ASRMODEL.weights.h5'
modelA_file_path= Path(modelA_file_path)

modelA_Config_file_path = './Models/ASRMODELCONFIG.json'
modelA_Config_file_path= Path(modelA_Config_file_path)



wav = Dpreprocess_wav(wav_path)
encoder = VoiceEncoder("cpu", weights_fpath=modelD_file_path)
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)


from spectralcluster import SpectralClusterer
from spectralcluster.utils import EigenGapType

clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=10,
    eigengap_type=EigenGapType.Ratio
)
labels = clusterer.predict(cont_embeds)


import json
# import numpy as np

def create_labelling(labels, wav_splits):
    
    from Diarization import sampling_rate
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0

    for i, time in enumerate(times):
        if i > 0 and labels[i] != labels[i - 1]:
            temp = {
                "start": float(start_time),
                "end": float(time),
                "speaker": int(labels[i - 1])
            }
            labelling.append(temp)
            start_time = time
        if i == len(times) - 1:
            temp = {
                "start": float(start_time),
                "end": float(time),
                "speaker": int(labels[i])
            }
            labelling.append(temp)

    return labelling

predicted_data = create_labelling(labels, wav_splits)

# Save the labelling to a JSON file
with open('./Output/Ourlabelling.json', 'w', encoding='utf-8') as f:
    json.dump(predicted_data, f, ensure_ascii=False, indent=4)

print("Labelling saved to labelling.json")


