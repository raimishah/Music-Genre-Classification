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


# define a PCA function to use (Lecture 5 - Slide 17)
# COV = (1/K)(X - X_mean).(X - X_mean)^T
def PCA(X,n): 
    X_2 = X - np.mean(X,axis=1,keepdims=True)                               # removed the mean
    COV = (X_2.dot(X_2.T))/(X_2.shape[1]-1)                                 # computed the covariance matrix  
    eigenvalues, eigenvecs = scipy.sparse.linalg.eigsh(COV,k=n)             # Got the eigenvectors and eigenvalues
    W = np.diag(1./(np.sqrt(eigenvalues))).dot(eigenvecs.T)                 # Lecture 5 - Slide 18 --> W = diag(eigenvalues^-1)*U^T
    return W

def ICA(X):  
    I_2 = np.identity(X.shape[0])
    W = np.identity(X.shape[0])
    const = 0.0001
    N = X.shape[1]
    for i in range(500):
        Y = W.dot(X)
        delta_w = (N*I_2-2*np.tanh(Y).dot(Y.T)).dot(W)
        W += const*delta_w

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

# function to compute classification accuracy
def classical_accuracy(pred,Y_test):
    test_ind = np.where(Y_test == 1)[0] # get the indices of data that has classical label
    pred_num = np.sum(pred[test_ind] == Y_test[test_ind]) #number of correct predictions
    acc = (pred_num / test_ind.shape[0])*100
    return acc
def one_second_result(probs,fps):
    probs = np.convolve(probs,np.ones(fps))
    return probs
# Accuracy function to compute classifier accuracy 
def get_acc(pred, y_test):
    return np.sum(y_test==pred)/len(y_test)*100
'''

classical_path = 'genres/classical'
metal_path = 'genres/metal'
blues_path = 'genres/blues'
pop_path = 'genres/pop'
country_path = 'genres/country'
disco_path = 'genres/disco'


data_classical = []
data_metal = []
data_blues = []
data_pop = []
data_country = []
data_disco = []
sampling_rate = 22050
num_tracks = 100

# First, load the classical music dataset
for file in os.listdir(classical_path):
    path = os.path.join(classical_path,file)
    fs,classical = wavread(path)
    classical = classical.astype(float)
    #classical = np.pad(classical, (0, 700000 - classical.shape[0]), 'constant', constant_values=(1, 1))
    shape = classical.shape[0] - 400000
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
    shape = metal.shape[0] - 400000
    metal = metal[:-shape]
    #print(metal.shape)
    data_metal.append(metal)

for file in os.listdir(blues_path):
    path = os.path.join(blues_path,file)
    fs,blues = wavread(path)
    blues = blues.astype(float)
    #blues = np.pad(blues, (0, 700000 - blues.shape[0]), 'constant', constant_values=(1, 1))
    shape = blues.shape[0] - 400000
    blues = blues[:-shape]
    #print(blues.shape)
    data_blues.append(blues)

for file in os.listdir(pop_path):
    path = os.path.join(pop_path,file)
    fs,pop = wavread(path)
    pop = pop.astype(float)
    #pop = np.pad(pop, (0, 700000 - pop.shape[0]), 'constant', constant_values=(1, 1))
    shape = pop.shape[0] - 400000
    pop = pop[:-shape]
    #print(pop.shape)
    data_pop.append(pop)

for file in os.listdir(country_path):
    path = os.path.join(country_path,file)
    fs,country = wavread(path)
    country = country.astype(float)
    #country = np.pad(country, (0, 700000 - country.shape[0]), 'constant', constant_values=(1, 1))
    shape = country.shape[0] - 400000
    country = country[:-shape]
    #print(country.shape)
    data_country.append(country)

for file in os.listdir(disco_path):
    path = os.path.join(disco_path,file)
    fs,disco = wavread(path)
    disco = disco.astype(float)
    #disco = np.pad(disco, (0, 700000 - disco.shape[0]), 'constant', constant_values=(1, 1))
    shape = disco.shape[0] - 400000
    disco = disco[:-shape]
    #print(disco.shape)
    data_disco.append(disco)


data_classical = np.array(data_classical)
data_metal = np.array(data_metal)
data_blues = np.array(data_blues)
data_pop = np.array(data_pop)
data_country = np.array(data_country)
data_disco = np.array(data_disco)

print('Musical data read... Starting spectogram computations...')

# Get spectogram features with log and magnitude
f,t,Zxx_classical = signal.stft(data_classical, fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx_classical = np.abs(Zxx_classical)
Zxx_classical = np.log(Zxx_classical)

f,t,Zxx_metal = signal.stft(data_metal, fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx_metal = np.abs(Zxx_metal)
Zxx_metal = np.log(Zxx_metal)

f,t,Zxx_blues = signal.stft(data_blues, fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx_blues = np.abs(Zxx_blues)
Zxx_blues = np.log(Zxx_blues)

f,t,Zxx_pop = signal.stft(data_pop, fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx_pop = np.abs(Zxx_pop)
Zxx_pop = np.log(Zxx_pop)

f,t,Zxx_country = signal.stft(data_country, fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx_country = np.abs(Zxx_country)
Zxx_country = np.log(Zxx_country)

f,t,Zxx_disco = signal.stft(data_disco, fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx_disco = np.abs(Zxx_disco)
Zxx_disco = np.log(Zxx_disco)

print('Done with spectogram computations...')

mix = np.arange(0, num_tracks, 1) # for random sampling of data
np.random.shuffle(mix)

Train_classical = np.hstack(Zxx_classical[mix[10:]]) # random 90% of classical data sampled
Train_metal = np.hstack(Zxx_metal[mix[10:]]) # random 90% of metal data sampled
Train_blues = np.hstack(Zxx_blues[mix[10:]]) # random 90% of metal data sampled
Train_pop = np.hstack(Zxx_pop[mix[10:]]) # random 90% of metal data sampled
Train_country = np.hstack(Zxx_country[mix[10:]]) # random 90% of metal data sampled
Train_disco = np.hstack(Zxx_disco[mix[10:]]) # random 90% of metal data sampled



Test_classical = np.hstack(Zxx_classical[mix[:10]]) # random remaining 10% is for testing
Test_metal = np.hstack(Zxx_metal[mix[:10]])
Test_blues = np.hstack(Zxx_blues[mix[:10]])
Test_pop = np.hstack(Zxx_pop[mix[:10]])
Test_country = np.hstack(Zxx_country[mix[:10]])
Test_disco = np.hstack(Zxx_disco[mix[:10]])



X_train = np.hstack((Train_classical,Train_metal,Train_blues,Train_pop,Train_country)) # stack'em
X_test = np.hstack((Test_classical,Test_metal,Test_blues,Test_pop,Test_country)) #stack'em

# Create labels for testing
# 1 is classical and 0 if metal
classical_labels = np.ones(Train_classical.shape[1])
metal_labels = np.zeros(Train_metal.shape[1])
blues_labels = np.ones(Train_blues.shape[1]) + np.ones(Train_blues.shape[1])
pop_labels = np.ones(Train_pop.shape[1]) + np.ones(Train_pop.shape[1]) + np.ones(Train_pop.shape[1])
country_labels = np.ones(Train_country.shape[1]) + np.ones(Train_country.shape[1]) + np.ones(Train_country.shape[1]) + np.ones(Train_country.shape[1])
disco_labels = np.ones(Train_disco.shape[1]) + np.ones(Train_disco.shape[1]) + np.ones(Train_disco.shape[1]) + np.ones(Train_disco.shape[1]) + np.ones(Train_disco.shape[1])


Y_train = np.hstack((classical_labels,metal_labels,blues_labels,pop_labels,country_labels))

classical_labels = np.ones(Test_classical.shape[1])
metal_labels = np.zeros(Test_metal.shape[1])
blues_labels = np.ones(Test_blues.shape[1]) + np.ones(Test_blues.shape[1])
pop_labels = np.ones(Test_pop.shape[1]) + np.ones(Test_pop.shape[1]) + np.ones(Test_pop.shape[1])
country_labels = np.ones(Test_country.shape[1]) + np.ones(Test_country.shape[1]) + np.ones(Test_country.shape[1]) + np.ones(Test_country.shape[1])
disco_labels = np.ones(Test_disco.shape[1]) + np.ones(Test_disco.shape[1]) + np.ones(Test_disco.shape[1]) + np.ones(Test_disco.shape[1]) + np.ones(Test_disco.shape[1])


Y_test = np.hstack((classical_labels,metal_labels,blues_labels,pop_labels,country_labels))

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
'''

print('Loading train and test data...')
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

X_train = try_to_load_as_pickled_object_or_None('X_train.pkl')
Y_train = try_to_load_as_pickled_object_or_None('Y_train.pkl')
X_test = try_to_load_as_pickled_object_or_None('X_test.pkl')
Y_test = try_to_load_as_pickled_object_or_None('Y_test.pkl')



print('Starting PCA...')
W = PCA(X_train,128)
# Uncomment below to test NMF
#W = NMF(X_train,128)


'''
Z_classical = W.dot(Train_classical - np.mean(Train_classical,axis=1,keepdims=True))
gauss_classical = gauss_classifier(Z_classical)

Z_metal = W.dot(Train_metal - np.mean(Train_metal,axis=1,keepdims=True))
gauss_metal = gauss_classifier(Z_metal)

Z_blues = W.dot(Train_blues - np.mean(Train_blues,axis=1,keepdims=True))
gauss_blues = gauss_classifier(Z_blues)
'''

print('Multiplication after PCA...')
Z_train = W.dot(X_train - np.mean(X_train,axis=1,keepdims=True))
Z_test = W.dot(X_test - np.mean(X_test,axis=1,keepdims=True))

## Uncomment below to test ICA ##
print('Using ICA on the spectogram PCA...')
#W_ica = ICA(Z_train)
#Z_train = W_ica.dot(Z_train)
#Z_test = W_ica.dot(Z_test)
##

print('Computing likelihoods for each data point...')
# Compute likelihoods for classical and music
'''
classical_probs = log_prob(Z_test,gauss_classical)
metal_probs = log_prob(Z_test,gauss_metal)
blues_probs = log_prob(Z_test,gauss_blues)
'''
G = [gauss_classifier(Z_train[:,Y_train == j]) for j in [0,1,2,3,4]]
probs = [log_prob(Z_test, i) for i in G]
#probs = one_second_result(probs,fps=26)
pred = np.argmax(probs, axis=0)
print('The accuracy of the classifier is: %f' %(get_acc(pred,Y_test)))

# According to Piazza, we get the 1-sec probs
'''
classical_probs = one_second_result(classical_probs,fps=26)
metal_probs = one_second_result(metal_probs,fps=26)
blues_probs = one_second_result(blues_probs,fps=26)
probs = np.hstack((classical_probs,metal_probs,blues_probs))
# get predictions
metal_pred = np.argmax(metal_probs)
'''

'''
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


'''












