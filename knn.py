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
from sklearn.neighbors import KNeighborsClassifier


def nearest_neighbor(X_train,Y_train,X_test,Y_test):
	
	knn = KNeighborsClassifier(n_neighbors = 2)
	knn.fit(X_train,Y_train)
	print("KNN Training Score: {:.3f}".format(knn.score(X_train,Y_train)))
	print("KNN Test score: {:.3f}".format(knn.score(X_test,Y_test)))
	