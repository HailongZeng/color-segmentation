import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def softmax_loss_matrix(X, y, C, K, alpha):
    '''
    Using matrix
    N:sample size     D:dimension, feature size    C:# of class
    :param X: shape:[N, D], training data
    :param y: shape:[N, 1], label
    :param C: number of classes
    :param K: iteration
    :param alpha: learning rate
    :return:
    '''
    N, D = X.shape
    y = np.asarray(y, dtype=np.int)
    w = np.zeros((D, C))
    b = np.zeros((1, C))
    for i in range(K):
        loss = 0.0
        s = X.dot(w) + b
        score = s - np.max(s, axis=1, keepdims=True)
        score_E = np.exp(score)
        Z = np.sum(score_E, axis=1, keepdims=True) #axis=1 rule following row
        prob = score_E / Z
        y_true_class = np.zeros_like(prob)
        y_true_class[range(N), y.reshape(N)] = 1.0  #one-hot encoding
        loss += -np.sum(y_true_class * np.log(prob)) / N
        dw = -np.dot(X.T, y_true_class - prob) / N
        db = -np.sum(prob - y_true_class, axis=0) / N
        w = w - alpha*dw
        b = b - alpha*db
        if i % 100 == 0:
            print(f'Iteration:{i}, loss:{np.round(loss, 4)}')
    return w, b
t1 = cv.getTickCount()  #Start time

X1 = np.loadtxt('Trainset1_RGB_softmax_(blue_barrel).txt')
X2 = np.loadtxt('Trainset1_RGB_softmax_(blue_not barrel).txt')
X3 = np.loadtxt('Trainset2_RGB.txt')
X4 = X3[1:1000000]
X = np.vstack((X1, X2, X4))
print(X1.shape)
print(X2.shape)
print(X4.shape)
print(X.shape)
y1 = np.zeros((X1.shape[0], 1))
y2 = np.ones((X2.shape[0], 1))
y3 = 2*np.ones((X4.shape[0], 1))
y = np.vstack((y1, y2, y3))
print(y1.shape)
print(y2.shape)
print(y3.shape)
print(y.shape)
del X1,X2,X3,X4,y1,y2,y3


K = 2000
alpha = 0.0001
C = 3

w, b = softmax_loss_matrix(X, y, C, K, alpha)

print(w)
print(b)
np.savetxt('w_{}_{}_RGB_Softmax_1.txt'.format(alpha, K), w)
np.savetxt('b_{}_{}_RGB_Softmax_1.txt'.format(alpha, K), b)

t2 = cv.getTickCount()  #Stop time
time = (t2-t1)/cv.getTickFrequency()
print('Time:%s s'%time)   #Spending time
