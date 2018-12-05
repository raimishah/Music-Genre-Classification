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
from sklearn.ensemble import RandomForestClassifier
from gaussian import get_gaussian
from knn import nearest_neighbor
from SVM import support_vector_machine
from confusion_matrix import confusion
import sklearn



# define a PCA function to use (Lecture 5 - Slide 17)
# COV = (1/K)(X - X_mean).(X - X_mean)^T
def PCA(X,n): 
    X_2 = X - np.mean(X,axis=1,keepdims=True)                               # removed the mean
    COV = (X_2.dot(X_2.T))/(X_2.shape[1]-1)                                 # computed the covariance matrix  
    eigenvalues, eigenvecs = scipy.sparse.linalg.eigsh(COV,k=n)             # Got the eigenvectors and eigenvalues
    W = np.diag(1./(np.sqrt(eigenvalues))).dot(eigenvecs.T)                 # Lecture 5 - Slide 18 --> W = diag(eigenvalues^-1)*U^T
    return W

def NMF(X,n): 
    Dim1 = X.shape[0]
    Dim2 = X.shape[1]
    H = np.random.rand(n,Dim2)
    W = np.random.rand(Dim1,n) # so that X = W.H
    for i in range(100): 
        H = H * ((W.T.dot(X)) / (0.000001 + W.T.dot(W).dot(H))) # prevent division by 0
        W = W * ((X.dot(H.T)) / (0.000001 + W.dot(H).dot(H.T)))
    return W

def one_second_result(probs,fps):
    probs = np.convolve(probs,np.ones(fps))
    return probs
# Accuracy function to compute classifier accuracy 
def get_acc(pred, y_test):
    return np.sum(y_test==pred)/len(y_test)*100


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj

############################################################################################################################################################################
GENRES=['METAL', 'CLASSICAL', 'BLUES', 'POP', 'COUNTRY']

print('Loading train and test data...')
X_train = try_to_load_as_pickled_object_or_None('X_train.pkl')
Y_train = try_to_load_as_pickled_object_or_None('Y_train.pkl')
X_test = try_to_load_as_pickled_object_or_None('X_test.pkl')
Y_test = try_to_load_as_pickled_object_or_None('Y_test.pkl')

print('Dimensionality reduction..')
W = PCA(X_train,64)
# Uncomment below to test NMF
#W = NMF(X_train,128)
Z_train = W.dot(X_train - np.mean(X_train,axis=1,keepdims=True))
Z_test = W.dot(X_test - np.mean(X_test,axis=1,keepdims=True))


# Gaussian Classifier
pred = get_gaussian(Z_train,Y_train,Z_test)
print('The accuracy of the gaussian classifier is: %f' %(get_acc(pred,Y_test)))
confusion(Y_test,pred)

# KNN
#nearest_neighbor(Z_train,Y_train,Z_test,Y_test)

# SVM 
#support_vector_machine(Z_train,Y_train,Z_test,Y_test)










