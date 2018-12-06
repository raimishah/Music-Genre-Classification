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
from sklearn.cluster import KMeans as sklearn_KMeans


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
        print(iters)
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

def lib_KMeans(X, k, seed):
    return sklearn_KMeans(n_clusters=k).fit(X)