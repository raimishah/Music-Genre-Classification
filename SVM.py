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
from sklearn.multiclass import OneVsRestClassifier
import os
from PIL import Image
import pickle
import sys
import librosa
from sklearn.svm import SVC

def get_acc(pred, y_test):
    return np.sum(y_test==pred)/len(y_test)*100

def support_vector_machine(X_train,Y_train,X_test,Y_test):
	svm = SVC()
	svm=SVC(C=3,kernel='rbf')
	svm.fit(X_train,Y_train)
	pred_train = svm.predict(X_train)
	pred_test = svm.predict(X_test)
	print(pred_test)
	print(Y_test)
	print("SVM Training Score: {:.3f}".format(get_acc(pred_train,Y_train)))
	print("SVM Test score: {:.3f}".format(get_acc(pred_test,Y_test)))
	return svm