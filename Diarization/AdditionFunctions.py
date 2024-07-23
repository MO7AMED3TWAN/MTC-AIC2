from Diarization.hparams import sampling_rate
from spectralcluster import SpectralClusterer
from spectralcluster.utils import EigenGapType


def create_labelling(labels, wav_splits):
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