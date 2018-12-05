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
from sklearn.svm import SVC


def support_vector_machine(X_train,Y_train,X_test,Y_test):
	svm = SVC(C = 100,gamma=0.08)
	svm.fit(X_train.transpose(),Y_train)
	print("SVM Training Score: {:.3f}".format(svm.score(X_train.transpose(),Y_train)))
	print("SVM Test score: {:.3f}".format(svm.score(X_test.transpose(),Y_test)))