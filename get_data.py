# import the necessary package dependencies
import numpy as np
import scipy
#from sklearn.model_selection import train_test_split
from scipy.io.wavfile import read as wavread
from scipy import signal
from scipy.fftpack import fft, fftfreq, ifft
import scipy.io
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
import sys
import librosa
import librosa.display


genres_paths = ['genres/metal',  'genres/classical', 'genres/blues', 'genres/pop', 'genres/country']

data = []
for g_path in genres_paths:
    show_spect = True
    for file in os.listdir(g_path):
        path = os.path.join(g_path,file)
        if '.wav' not in path:
            continue
        y, sr = librosa.load(path, mono=True)
        S = librosa.feature.melspectrogram(y, sr=sr,n_mels=128,n_fft=2048,hop_length=1024).T
        if show_spect:
            librosa.display.specshow(librosa.power_to_db(S.T, ref = np.max), y_axis='mel', fmax=8000, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram')
            plt.tight_layout()
            plt.show()
            show_spect = False
        S = librosa.power_to_db(S,ref=np.max)
        S = S[:-1 * (S.shape[0] % 128)]
        #num_chunk   = S.shape[0] / 128
        #data_chunks = np.split(S, num_chunk)
        S = librosa.feature.mfcc(S=S,sr=sr)
        #shape = S.shape[0] - 640
        #S = S[:-shape,:]
        data.append(S)

data = np.array(data)
labels = np.zeros(data.shape[0])
for i in range(0, 10):
    labels[i*100 : (i + 1) * 100] = i
print(data.shape)


def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    print('Saving data...')
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

save_as_pickled_object(data.reshape(500,20*128),'data.pkl')
save_as_pickled_object(labels,'labels.pkl')

