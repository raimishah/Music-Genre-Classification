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

def gauss_classifier(X):
        N = X.shape[1]
        # get mean and covariances of data
        Mean = np.mean(X,axis=1,keepdims=True)
        Cov = (X-Mean).dot((X-Mean).T)
        Cov /= (N-1)
        # return both of them as a dict 
        g_dict = {'mean':Mean, 'cov':Cov}
        return g_dict

def log_prob(X,g):
    M = X.shape[0]
    X_2 = X - g['mean']
    IC = np.linalg.pinv(g['cov'])
    probs = (-1*np.log(np.linalg.det(IC)) + M*np.log(2*np.pi) + np.sum((IC).dot(X_2)*X_2,axis=0))*(-1/2)
    return probs


def get_gaussian(Z_train,Y_train,Z_test):
	G = [gauss_classifier(Z_train[:,Y_train == j]) for j in [0,1,2,3,4]]
	probs = [log_prob(Z_test, i) for i in G]
	pred = np.argmax(probs, axis=0)
	return pred