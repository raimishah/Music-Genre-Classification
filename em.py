import scipy
import matplotlib
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import sklearn.preprocessing
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans as SKMeans



#make some testing data.
m0 = np.asarray([-1,-1])
m1 = np.asarray([1,1])
m2 = np.asarray([3,3])

pts = np.zeros((300, 2))
for i in range(0, 100):
    pts[i] = m0 + np.random.normal(0, .5, (m0.shape))
    plt.scatter(pts[i][0], pts[i][1], color = 'r')
for i in range(100, 200):
    pts[i] = m1 + np.random.normal(0, .2, (m0.shape))
    plt.scatter(pts[i][0], pts[i][1], color = 'g')
for i in range(200, 300):
    pts[i] = m2 + np.random.normal(0, .6, (m0.shape))
    plt.scatter(pts[i][0], pts[i][1], color = 'b')

plt.show()


num_classes = 3
data = pts


def multivariateGaussian(X, mu, cov):
        return (1/np.sqrt(np.linalg.det(2*np.pi*cov))) * np.exp(-.5 * ((X - mu).T @ np.linalg.pinv(cov) @ (X - mu)))

def E_step(X, mus, covs, pis):
    N = X.shape[0]
    K = pis.shape[0]
    posteriors = np.zeros((K, N))
    for k in range(K):
        for n in range(N):
            posteriors[k, n] = pis[k] * multivariateGaussian(X[n], mus[k], covs[k])
    posteriors /= posteriors.sum(0)
    return posteriors

def M_step(X, mus, covs, pis, posteriors):
    N = X.shape[0]
    K = pis.shape[0]
    for k in range(K):
        N_k = np.sum(posteriors[k, :])
        pis[k] = N_k / N

        mus[k, :] = 0
        for n in range(N):
            mus[k, :] = posteriors[k, n] * X[n]
        mus[k, :] /= N_k

        covs[k, :] = 0
        for n in range(N):
            covs[k, :] = posteriors[k, n] * ((X[n] - mus[k]) @ (X[n] - mus[k]).T)
        covs[k, :] /= N_k
    return mus, covs, pis


print(data.shape)
mus = np.random.randn(num_classes, data.shape[1])
covs = []
for i in range(num_classes):
    covs.append(np.eye(data.shape[1]))
covs = np.array(covs)
pis = np.random.uniform(0, 1, (num_classes, ))


for i in range(5):
    posteriors = E_step(data, mus, covs, pis)
    mus, covs, pis = M_step(data, mus, covs, pis, posteriors)

    #plot
    for j in range(posteriors.shape[1]):
        if np.argmax(posteriors[:,j]) == 0:
            plt.scatter(pts[j][0], pts[j][1], color = 'r')
        elif np.argmax(posteriors[:,j]) == 1:
            plt.scatter(pts[j][0], pts[j][1], color = 'g')
        elif np.argmax(posteriors[:,j]) == 2:
            plt.scatter(pts[j][0], pts[j][1], color = 'b')

    plt.show()
