import glob
import numpy as np
import librosa
import pandas as pd
import wave
import os
from scipy.io.wavfile import read
from matplotlib import pyplot as plt

from pymongo import MongoClient
client = MongoClient()
db = client['audio']
tracks = db.tracks

def cmn(vec, variance_normalization=False):
    """ Cepstral Mean Normalization
        Reduce the noise across the frequency spectrum by taking the mean
        on axis 0, and subtracting each row from the norm.  This raises
        the decibel level of the harmonics and reduces the noise.
    """
    eps = 2**-30
    rows, cols = vec.shape

    norm = np.mean(vec, axis=0)
    norm_vec = np.tile(norm, (rows, 1))

    return vec - norm_vec


def gen_mel(gender):
    """ Load the data from the DSD100 and ccmixter from MongoDB
        Split into 20 second chunks.  The sample rate should always be 48000.
        Generate Mel Spectrogram
        Apply power_to_db to put the frequencies on a log-dB scale
        Apply Cepstral Mean Normalization
        Save it out to the new path and update MongoDB
    """
    for track in tracks.find({'component':'vocals.wav', 'gender':gender}):
        suffix = os.path.splitext(track['coch'].split('vocals-')[1])[0]
        y, sr = librosa.load(track['path'])
        y = y[y != 0]
        for i in range(1, 100):
            # stop when the subset exceeds the sample rate
            if sr * 20 * i > len(y[:i*20*sr]):
                break
            mel_path = f"data/{gender}/{suffix}-{track['_id']}-mel-{i}.txt"

            y2 = y[sr*20*(i-1):sr*20*i]
            S = librosa.feature.melspectrogram(y=y2, sr=sr, n_fft=2048, hop_length=512)
            S_dB = librosa.power_to_db(S, ref=np.max)
            norm_S_dB = cmn(S_dB)

            np.savetxt(mel_path, norm_S_dB)
            track['mel_path'] = mel_path
            tracks.replace_one({'_id': track['_id']}, track)
