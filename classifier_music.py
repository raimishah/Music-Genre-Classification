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


# define a PCA function to use (Lecture 5 - Slide 17)
# COV = (1/K)(X - X_mean).(X - X_mean)^T
def PCA(X,n): 
	X_2 = X - np.mean(X,axis=1,keepdims=True)    							# removed the mean
	COV = (X_2.dot(X_2.T))/(X_2.shape[1]-1)        							# computed the covariance matrix  
	eigenvalues, eigenvecs = scipy.sparse.linalg.eigsh(COV,k=n)  		    # Got the eigenvectors and eigenvalues
	W = np.diag(1./(np.sqrt(eigenvalues))).dot(eigenvecs.T)                 # Lecture 5 - Slide 18 --> W = diag(eigenvalues^-1)*U^T
	return W

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

classical_path = 'genres/classical'
metal_path = 'genres/metal'


data_classical = []
data_metal = []
sampling_rate = 22050
num_tracks = 100

# First, load the classical music dataset
for file in os.listdir(classical_path):
    path = os.path.join(classical_path,file)
    fs,classical = wavread(path)
    classical = classical.astype(float)
    #classical = np.pad(classical, (0, 700000 - classical.shape[0]), 'constant', constant_values=(1, 1))
    shape = classical.shape[0] - 500000
    #print(shape)
    classical = classical[:-shape]
    #print(classical.shape)
    data_classical.append(classical)

# Next, load the music dataset    
for file in os.listdir(metal_path):
    path = os.path.join(metal_path,file)
    fs,metal = wavread(path)
    metal = metal.astype(float)
    #metal = np.pad(metal, (0, 700000 - metal.shape[0]), 'constant', constant_values=(1, 1))
    shape = metal.shape[0] - 500000
    metal = metal[:-shape]
    #print(metal.shape)
    data_metal.append(metal)

data_classical = np.array(data_classical)
data_metal = np.array(data_metal)
print(data_metal.shape,data_classical.shape)

# Get spectogram features with log and magnitude
f,t,Zxx_classical = signal.stft(data_classical, fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx_classical = np.abs(Zxx_classical)
Zxx_classical = np.log(Zxx_classical)

f,t,Zxx_metal = signal.stft(data_metal, fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx_metal = np.abs(Zxx_metal)
Zxx_metal = np.log(Zxx_metal)

mix = np.arange(0, num_tracks, 1) # for random sampling of data
np.random.shuffle(mix)

Train_classical = np.hstack(Zxx_classical[mix[10:]]) # random 90% of classical data sampled
Train_metal = np.hstack(Zxx_metal[mix[10:]]) # random 90% of metal data sampled
Test_classical = np.hstack(Zxx_classical[mix[:10]]) # random remaining 10% is for testing
Test_metal = np.hstack(Zxx_metal[mix[:10]])

X_train = np.hstack((Train_classical,Train_metal)) # stack'em
X_test = np.hstack((Test_classical,Test_metal)) #stack'em

# Create labels for testing
# 1 is classical and 0 if metal
classical_labels = np.ones(Test_classical.shape[1])
metal_labels = np.zeros(Test_metal.shape[1])
Y_test = np.hstack((classical_labels,metal_labels))

W = PCA(X_train,64)

Z_classical = W.dot(Train_classical - np.mean(Train_classical,axis=1,keepdims=True))
gauss_classical = gauss_classifier(Z_classical)

Z_metal = W.dot(Train_metal - np.mean(Train_metal,axis=1,keepdims=True))
gauss_metal = gauss_classifier(Z_metal)

# function to compute classification accuracy
def classical_accuracy(pred,Y_test):
    test_ind = np.where(Y_test == 1)[0] # get the indices of data that has classical label
    pred_num = np.sum(pred[test_ind] == Y_test[test_ind]) #number of correct predictions
    acc = (pred_num / test_ind.shape[0])*100
    return acc
def one_second_result(probs,fps):
    probs = np.convolve(probs,np.ones(fps))
    return probs

Z_test = W.dot(X_test - np.mean(X_test,axis=1,keepdims=True))
# Compute likelihoods for classical and music
classical_probs = log_prob(Z_test,gauss_classical)
metal_probs = log_prob(Z_test,gauss_metal)
probs = np.hstack((classical_probs,metal_probs))

# According to Piazza, we get the 1-sec probs
classical_probs = one_second_result(classical_probs,fps=26)
metal_probs = one_second_result(metal_probs,fps=26)
# get predictions
result = (classical_probs > metal_probs).astype(int)


print('The accuracy of the classical classifier is: %f' %(classical_accuracy(result,Y_test)))

print('Testing with some random metal music...')
rate,data = wavread('metal.00023.wav') 
data = data.astype(float)
f,t,music = signal.stft(data, fs = rate, window = 'hann', nperseg = 1024, noverlap=768)
music = np.log(np.abs(music))
Z_music = W.dot(music - np.mean(music,axis=1,keepdims=True))
classical_probs = log_prob(Z_music,gauss_classical)
metal_probs = log_prob(Z_music,gauss_metal)

result = (classical_probs > metal_probs).astype(int)


if (classical_accuracy(result,np.ones(result.shape[0]))) < 50:
    print('I am pretty sure that this is METAL MUSIC!')
else: 
    print('I am pretty sure that this is CLASSICAL MUSIC!')

print('Testing with some random classical music')
rate,data = wavread('classical.00009.wav') 
data = data.astype(float)
f,t,music = signal.stft(data, fs = rate, window = 'hann', nperseg = 1024, noverlap=768)
music = np.log(np.abs(music))
Z_music = W.dot(music - np.mean(music,axis=1,keepdims=True))
classical_probs = log_prob(Z_music,gauss_classical)
metal_probs = log_prob(Z_music,gauss_metal)

result = (classical_probs > metal_probs).astype(int)


if (classical_accuracy(result,np.ones(result.shape[0]))) < 50:
    print('I am pretty sure that this is METAL MUSIC!')
else: 
    print('I am pretty sure that this is CLASSICAL MUSIC!')















