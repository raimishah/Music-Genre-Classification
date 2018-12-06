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
from sklearn.model_selection import train_test_split
from gaussian import get_gaussian
from knn import nearest_neighbor
from SVM import support_vector_machine
from KMeans import KMeans
from KMeans import lib_KMeans
from confusion_matrix import confusion
import sklearn
from sklearn import mixture
import pandas as pd
#from train import train_net
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from train import train_torch




# define a PCA function to use (Lecture 5 - Slide 17)
# COV = (1/K)(X - X_mean).(X - X_mean)^T
def PCA(X, n):
    # getting 0 mean
    X_ = X - np.mean(X, axis = 1, keepdims = True)
    cov_X = (X_ @ X_.T) / (X.shape[1] - 1)
    evals, evecs = scipy.sparse.linalg.eigsh(cov_X, n)
    evals = np.sqrt(evals)
    evals = 1 / evals
    evals = np.diag(evals)
    W = evals.dot(evecs.T)
    Z = W.dot(X_)    
    return W, Z

def NMF(X,n): 
    Dim1 = X.shape[0]
    Dim2 = X.shape[1]
    H = np.random.rand(n,Dim2)
    W = np.random.rand(Dim1,n) # so that X = W.H
    for i in range(100): 
        H = H * ((W.T.dot(X)) / (0.000001 + W.T.dot(W).dot(H))) # prevent division by 0
        W = W * ((X.dot(H.T)) / (0.000001 + W.dot(H).dot(H.T)))
    return W, W@H

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

print('Loading train and test data...')
X_all = try_to_load_as_pickled_object_or_None('data.pkl')
Y_all = try_to_load_as_pickled_object_or_None('labels.pkl')


print('Dimensionality reduction..')

X_all = X_all.T
print(X_all.shape)
W,Z_p = PCA(X_all, 256)
Z_p = Z_p.T
print(Z_p.shape)

X_train, X_test, Y_train, Y_test = train_test_split(Z_p, Y_all, test_size = 0.1)

# Uncomment below to test NMF
#W, Z_p = NMF(X_all.T,16)
#Z_p = W.dot(X_all - np.mean(X_all,axis=1,keepdims=True))
#print(Z_p.shape)
#X_train, X_test, Y_train, Y_test = train_test_split(Z_p, Y_all, test_size = 0.1)

#first we need to map colors on labels
#plot for 2d
#plt.scatter(Z_train[1, :], Z_train[0, :], color = 'r')
#plt.show()

#plot for 2d
'''
for i in range(len(Y_all)):
    if Y_all[i] == 0:
        plt.scatter(Z_p[i, 1], Z_p[i, 0], color = 'r')
    elif Y_all[i] == 1:
        plt.scatter(Z_p[i, 1], Z_p[i, 0], color = 'b')
    elif Y_all[i] == 2:
        plt.scatter(Z_p[i, 1], Z_p[i, 0], color = 'g')
    elif Y_all[i] == 3:
        plt.scatter(Z_p[i, 1], Z_p[i, 0], color = 'k')
    elif Y_all[i] == 4:
        plt.scatter(Z_p[i, 1], Z_p[i, 0], color = 'c')
'''
#plt.show()

# Gaussian Classifier
print(X_train.shape)
print(Y_train.shape)
print(Y_test)
pred = get_gaussian(X_train.T,Y_train,X_test.T)
print('The accuracy of the gaussian classifier is: %f' %(get_acc(pred,Y_test)))
#confusion(Y_test,pred)


#train_net(X_train.T,Y_train,X_test.T,Y_test)

#plot for 3d
fig = plt.figure()
ax = Axes3D(fig)

for i in range(len(Y_all)):
    if Y_all[i] == 0:
        ax.scatter(Z_p[i, :][0], Z_p[i, :][1], Z_p[i, :][2], c = 'b', marker='o')
    elif Y_all[i] == 1:
        ax.scatter(Z_p[i, :][0], Z_p[i, :][1], Z_p[i, :][2], c = 'r', marker='o')
    elif Y_all[i] == 2:
        ax.scatter(Z_p[i, :][0], Z_p[i, :][1], Z_p[i, :][2], c = 'g', marker='o')
    elif Y_all[i] == 3:
        ax.scatter(Z_p[i, :][0], Z_p[i, :][1], Z_p[i, :][2], c = 'c', marker='o')
    elif Y_all[i] == 4:
        ax.scatter(Z_p[i, :][0], Z_p[i, :][1], Z_p[i, :][2], c = 'k', marker='o')
#plt.show()

#KNN
nearest_neighbor(X_train,Y_train,X_test,Y_test)

# SVM 
support_vector_machine(X_train,Y_train,X_test,Y_test)

# Neural-Net
train_torch(X_train,Y_train,X_test,Y_test)

#KMeans
for i in range(10):
    kmeans = lib_KMeans(Z_p, 5, 0)
    #print(Y_all)
    #print(kmeans.labels_)
    #randomly sample idxs to evaluate on - fowlkes mallows doesnt work on whole list
    randomness = np.arange(0, int(len(Y_all) / 1.2))
    np.random.shuffle(randomness)
    Y_clustering_test = Y_all[randomness]
    Y_clustering_pred = kmeans.labels_[randomness]
    fowlkes_scores = metrics.fowlkes_mallows_score(Y_clustering_test, Y_clustering_pred)
    print("KMeans accuracy : " + str(fowlkes_scores))


    #sklearn GMM
    gmm = mixture.GaussianMixture(n_components=5, covariance_type='spherical', n_init=5).fit(X_train)
    gmm_pred = gmm.predict(X_test)
    fowlkes_scores = metrics.fowlkes_mallows_score(Y_test, gmm_pred)
    print('gmm accuracy : ' + str(fowlkes_scores))








