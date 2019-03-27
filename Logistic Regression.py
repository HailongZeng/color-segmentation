import numpy as np
import cv2 as cv

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def gradDescent(alpha, Iter, X, y):
    m, n = X.shape  #sample size
    w = np.zeros((n,1))  #initialization
    b = np.zeros((1,1))   #initialization
    w = np.mat(w)        #transformed to matrix
    b = np.mat(b)        #transformed to matrix
    print(X.transpose())
    for k in range(Iter):
        h = sigmoid(np.dot(X, w) + b)
        error = (y-h)

        #update
        w = w + alpha*(np.dot(X.T,error))
        b = b + alpha*(np.sum(error.T))
    return w, b

t1 = cv.getTickCount()  #Start time
Trainset1 = np.loadtxt('Trainset1_RGB_1.txt')
Trainset2 = np.loadtxt('Trainset2_RGB.txt')
X1 = np.mat(Trainset1)  #class blue, p(y=1|x) = exp(xw)/(1+exp(xw))
X2 = np.mat(Trainset2)  #class not blue, p(y=0|x) = 1/(1+exp(xw))
del Trainset1
del Trainset2

X = np.vstack((X1, X2))  #concatenate all the data from all classes to have training data
m1, n1 = X1.shape  #m1 is row size, sample size;n1 is column size, dimension
m2, n2 = X2.shape  #m2 is row size, sample size;n2 is column size, dimension
X = X/255
print(X)
y1 = np.ones((m1,1))
y2 = np.zeros((m2,1))
y = np.vstack((y1, y2))
y = np.mat(y)   #transformed to matrix

alpha = 0.00001
K = 1000
w, b = gradDescent(alpha, K, X, y)
print(w)
print(b)
p1 = sigmoid(np.dot(X, w)+b)  #The probability of class blue, if >0.5, blue; <0.5, not blue
# p2 = 1.0/(1+np.exp(np.dot(X, w) + b))  #class not blue
np.savetxt('w_{}_{}_RGB.txt'.format(alpha, K), w)
np.savetxt('b_{}_{}_RGB.txt'.format(alpha, K), b)

t2 = cv.getTickCount()  #Stop time
time = (t2-t1)/cv.getTickFrequency()
print('Time:%s s'%time)   #Spending time

