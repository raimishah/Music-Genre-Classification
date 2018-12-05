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


classical_path = 'genres/classical'
metal_path = 'genres/metal'
blues_path = 'genres/blues'
pop_path = 'genres/pop'
country_path = 'genres/country'


data_classical = []
data_metal = []
data_blues = []
data_pop = []
data_country = []
sampling_rate = 22050
num_tracks = 100

# First, load the classical music dataset
for file in os.listdir(classical_path):
    path = os.path.join(classical_path,file)
    if '.wav' not in path:
        continue
    y, sr = librosa.load(path, mono=True)
    S = librosa.feature.melspectrogram(y, sr=sr,n_mels=128,n_fft=2048,hop_length=1024)
    S = librosa.power_to_db(S,ref=np.max)
    #S = librosa.feature.mfcc(S=S,sr=sr)
    shape = S.shape[1] - 640
    S = S[:,:-shape]
    data_classical.append(S)

# Next, load the music dataset    
for file in os.listdir(metal_path):
    path = os.path.join(metal_path,file)
    if '.wav' not in path:
        continue
    y, sr = librosa.load(path, mono=True)
    S = librosa.feature.melspectrogram(y, sr=sr,n_mels=128,n_fft=2048,hop_length=1024)
    S = librosa.power_to_db(S,ref=np.max)
    #S = librosa.feature.mfcc(S=S,sr=sr)
    shape = S.shape[1] - 640
    S = S[:,:-shape]
    data_metal.append(S)

for file in os.listdir(blues_path):
    path = os.path.join(blues_path,file)
    if '.wav' not in path:
        continue
    y, sr = librosa.load(path, mono=True)
    S = librosa.feature.melspectrogram(y, sr=sr,n_mels=128,n_fft=2048,hop_length=1024)
    S = librosa.power_to_db(S,ref=np.max)
    #S = librosa.feature.mfcc(S=S,sr=sr)
    shape = S.shape[1] - 640
    S = S[:,:-shape]
    data_blues.append(S)

for file in os.listdir(pop_path):
    path = os.path.join(pop_path,file)
    if '.wav' not in path:
        continue
    y, sr = librosa.load(path, mono=True)
    S = librosa.feature.melspectrogram(y, sr=sr,n_mels=128,n_fft=2048,hop_length=1024)
    S = librosa.power_to_db(S,ref=np.max)
    #S = librosa.feature.mfcc(S=S,sr=sr)
    shape = S.shape[1] - 640
    S = S[:,:-shape]
    data_pop.append(S)

for file in os.listdir(country_path):
    path = os.path.join(country_path,file)
    if '.wav' not in path:
        continue
    y, sr = librosa.load(path, mono=True)
    S = librosa.feature.melspectrogram(y, sr=sr,n_mels=128,n_fft=2048,hop_length=1024)
    S = librosa.power_to_db(S,ref=np.max)
    #S = librosa.feature.mfcc(S=S,sr=sr)
    shape = S.shape[1] - 640
    S = S[:,:-shape]
    data_country.append(S)


data_classical = np.array(data_classical)
data_metal = np.array(data_metal)
data_blues = np.array(data_blues)
data_pop = np.array(data_pop)
data_country = np.array(data_country)

mix = np.arange(0, num_tracks, 1) # for random sampling of data
np.random.seed()
np.random.shuffle(mix)

Train_classical = np.hstack(data_classical[mix[10:]]) # random 90% of classical data sampled
Train_metal = np.hstack(data_metal[mix[10:]]) # random 90% of metal data sampled
Train_blues = np.hstack(data_blues[mix[10:]]) # random 90% of metal data sampled
Train_pop = np.hstack(data_pop[mix[10:]]) # random 90% of metal data sampled
Train_country = np.hstack(data_country[mix[10:]]) # random 90% of metal data sampled



Test_classical = np.hstack(data_classical[mix[:10]]) # random remaining 10% is for testing
Test_metal = np.hstack(data_metal[mix[:10]])
Test_blues = np.hstack(data_blues[mix[:10]])
Test_pop = np.hstack(data_pop[mix[:10]])
Test_country = np.hstack(data_country[mix[:10]])


print(Train_classical.shape)
X_train = np.hstack((Train_classical,Train_metal,Train_blues,Train_pop,Train_country)) # stack'em
X_test = np.hstack((Test_classical,Test_metal,Test_blues,Test_pop,Test_country)) #stack'em

# Create labels for testing
metal_labels = np.zeros(Train_metal.shape[1])
classical_labels = np.ones(Train_classical.shape[1])
blues_labels = np.ones(Train_blues.shape[1]) + np.ones(Train_blues.shape[1])
pop_labels = np.ones(Train_pop.shape[1]) + np.ones(Train_pop.shape[1]) + np.ones(Train_pop.shape[1])
country_labels = np.ones(Train_country.shape[1]) + np.ones(Train_country.shape[1]) + np.ones(Train_country.shape[1]) + np.ones(Train_country.shape[1])


Y_train = np.hstack((metal_labels,classical_labels,blues_labels,pop_labels,country_labels))

metal_labels = np.zeros(Test_metal.shape[1])
classical_labels = np.ones(Test_classical.shape[1])
blues_labels = np.ones(Test_blues.shape[1]) + np.ones(Test_blues.shape[1])
pop_labels = np.ones(Test_pop.shape[1]) + np.ones(Test_pop.shape[1]) + np.ones(Test_pop.shape[1])
country_labels = np.ones(Test_country.shape[1]) + np.ones(Test_country.shape[1]) + np.ones(Test_country.shape[1]) + np.ones(Test_country.shape[1])


Y_test = np.hstack((metal_labels,classical_labels,blues_labels,pop_labels,country_labels))

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

save_as_pickled_object(X_train,'X_train.pkl')
save_as_pickled_object(Y_train,'Y_train.pkl')
save_as_pickled_object(X_test,'X_test.pkl')
save_as_pickled_object(Y_test,'Y_test.pkl')