import scipy
import matplotlib
from scipy.io import wavfile
from scipy import signal
from scipy.io.wavfile import read as wavread
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
import os

from sklearn.cluster import KMeans as SKMeans


#open all files and get all data

genres_paths = ['genres/blues',  'genres/classical', 'genres/country', 'genres/disco', 'genres/hiphop', 'genres/jazz', 'genres/metal', 'genres/pop', 'genres/reggae', 'genres/rock']

data = []
for g_path in genres_paths:
    for file in os.listdir(g_path):
        path = os.path.join(g_path,file)
        if '.wav' not in path:
            continue
        fs,temp_data = wavread(path)
        temp_data = temp_data.astype(float)
        shape = temp_data.shape[0] - 500000
        temp_data = temp_data[:-shape]
        data.append(temp_data)

data = np.array(data)
labels = np.zeros(data.shape[0])
for i in range(0, 10):
    labels[i*100 : (i + 1) * 100] = i
print(data.shape)


sampling_rate = 22050
Zxx_data = []
for i in range(len(genres_paths)):
#for i in range(0, 1):
    f,t,Zxx = signal.stft(data[i*100 : (i + 1) * 100], fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
    Zxx = np.abs(Zxx)
    Zxx = np.log(Zxx)
    Zxx_data.append(Zxx)

'''
f,t,Zxx = signal.stft(data[0 : 100], fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx = np.abs(Zxx)
Zxx = np.log(Zxx)
Zxx_data.append(Zxx)

f,t,Zxx = signal.stft(data[100 : 200], fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx = np.abs(Zxx)
Zxx = np.log(Zxx)
Zxx_data.append(Zxx)

f,t,Zxx = signal.stft(data[200 : 300], fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx = np.abs(Zxx)
Zxx = np.log(Zxx)
Zxx_data.append(Zxx)

f,t,Zxx = signal.stft(data[300 : 400], fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx = np.abs(Zxx)
Zxx = np.log(Zxx)
Zxx_data.append(Zxx)

f,t,Zxx = signal.stft(data[400 : 500], fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx = np.abs(Zxx)
Zxx = np.log(Zxx)
Zxx_data.append(Zxx)

f,t,Zxx = signal.stft(data[500 : 600], fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx = np.abs(Zxx)
Zxx = np.log(Zxx)
Zxx_data.append(Zxx)

f,t,Zxx = signal.stft(data[700 : 800], fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx = np.abs(Zxx)
Zxx = np.log(Zxx)
Zxx_data.append(Zxx)

f,t,Zxx = signal.stft(data[800 : 900], fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx = np.abs(Zxx)
Zxx = np.log(Zxx)
Zxx_data.append(Zxx)

f,t,Zxx = signal.stft(data[900 : 1000], fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx = np.abs(Zxx)
Zxx = np.log(Zxx)
Zxx_data.append(Zxx)
'''
print(Zxx.shape)

#pca
def PCA(X, n, plot_eigvecs):
    # getting 0 mean
    X_ = X - np.mean(X, axis = 1, keepdims = True)
    cov_X = (X_ @ X_.T) / (X.shape[1] - 1)
    evals, evecs = scipy.sparse.linalg.eigsh(cov_X, n)
    if plot_eigvecs:
        nums = np.arange(0, len(evecs), 1)
        norms = [np.linalg.norm(evecs[i, :]) for i in range(evecs.shape[0])]
        plt.plot(nums, sorted(norms, reverse=True))
        plt.yscale('log')
        plt.show()
    evals = np.sqrt(evals)
    evals = 1 / evals
    evals = np.diag(evals)
    W = evals.dot(evecs.T)
    Z = W.dot(X_)    
    return W, Z

def KMeans(X, k,seed):
    #pick random cluster centers
    centroids = np.zeros((k, X.shape[1]))
    #np.random.seed(0)
    np.random.seed(seed)
    center_idxs = np.random.choice(np.arange(0, X.shape[0]), size=k)
    #center_idxs = [10, X.shape[1] - 1, 950]

    for i in range(k):
        centroids[i] = X[center_idxs[i]]

    clusters = []
    old_assignments = np.empty((X.shape[0], ))
    iters = 0
    while(True):
        #assign each point to closest center
        dists = scipy.spatial.distance.cdist(X, centroids)
        assignments = np.argmin(dists, axis = 1)

        #check if converged
        if np.array_equal(old_assignments, assignments):
            break
        old_assignments = assignments
        
        #update centers
        for i in range(k):
            pts = np.where(assignments == i)
            pts = np.array(pts)
            pts = pts.T
            avg_array = np.zeros((centroids[i].shape[0], 1))
            for j in range(pts.shape[0]):
                avg_array += X[pts[j]].T
            avg_array = avg_array / pts.shape[0]
            centroids[i] = avg_array.T

        iters += 1
    return centroids, assignments


#PCA
W_p, Z_p = PCA(X, 64, False)

#KMeans
ccs, assignments = KMeans(Z_p, 10, 0)

#evaluate
unique, counts = np.unique(labels, return_counts=True)
dict_real = dict(zip(unique, counts))
print(dict_real)

unique, counts = np.unique(assignments, return_counts=True)
dict_pred = dict(zip(unique, counts))
print(dict_pred)





