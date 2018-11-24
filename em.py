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
    '''
    for k in range(K):
        for n in range(N):
            posteriors[k, n] = pis[k] * multivariateGaussian(X[n], mus[k], covs[k])

    posteriors /= posteriors.sum(0)
    '''

    for n in range(N):
        for k in range(K):
            numer = pis[k] * multivariateGaussian(X[n], mus[k], covs[k])
            denom = 0
            for j in range(K):
                denom += pis[j] * multivariateGaussian(X[n], mus[j], covs[j])

            posteriors[k, n] = numer / denom

    return posteriors

def M_step(X, mus, covs, pis, posteriors):
    '''
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
    '''
    n, p = X.shape
    k = len(pis)
    pis = np.zeros(k)

    
    for j in range(len(mus)):
        for i in range(n):
            pis[j] += posteriors[j, i]
    pis /= n

    mus = np.zeros((k, p))
    for j in range(k):
        for i in range(n):
            mus[j] += posteriors[j, i] * X[i]
        mus[j] /= posteriors[j, :].sum()

    sigmas = np.zeros((k, p, p))
    for j in range(k):
        for i in range(n):
            ys = np.reshape(X[i]- mus[j], (2,1))
            sigmas[j] += posteriors[j, i] * np.dot(ys, ys.T)
        sigmas[j] /= posteriors[j,:].sum()

    return mus, sigmas, pis



'''
mus = np.random.randn(num_classes, data.shape[1])
covs = []
for i in range(num_classes):
    covs.append(np.eye(data.shape[1]))
covs = np.array(covs)
pis = np.random.uniform(0, 1, (num_classes, ))
'''
'''
np.random.seed(0)
pis = np.random.random(num_classes)
pis /= pis.sum()
mus = np.random.random((num_classes,2))
covs = np.array([np.eye(data.shape[1])] * num_classes)


for i in range(2):
    print(pis)
    print(mus)
    print(covs)
    posteriors = E_step(data, mus, covs, pis)
    #mus, covs, pis = M_step(data, mus, covs, pis, posteriors)
    M_step(data, mus, covs, pis, posteriors)
    print(pis)
    print(mus)
    print(covs)

    #plot
    for j in range(posteriors.shape[1]):
        if np.argmax(posteriors[:,j]) == 0:
            plt.scatter(pts[j][0], pts[j][1], color = 'r')
        elif np.argmax(posteriors[:,j]) == 1:
            plt.scatter(pts[j][0], pts[j][1], color = 'g')
        elif np.argmax(posteriors[:,j]) == 2:
            plt.scatter(pts[j][0], pts[j][1], color = 'b')

    plt.show()
'''


#test kmeans
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


ccs, assignments = KMeans(data, 3, 0)
for j in range(len(assignments)):
    if assignments[j] == 0:
        plt.scatter(pts[j][0], pts[j][1], color = 'r')
    elif assignments[j] == 1:
        plt.scatter(pts[j][0], pts[j][1], color = 'g')
    if assignments[j] == 2:
        plt.scatter(pts[j][0], pts[j][1], color = 'b')

plt.show()
