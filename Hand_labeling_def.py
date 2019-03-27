from roipoly import RoiPoly
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import cv2 as cv
import os
def Hand_labeling(img):
    matplotlib.use('TkAgg')

    plt.imshow(img)
    plt.show(block=False)

    # Draw ROI1
    my_roi1 = RoiPoly(color = 'b')  # draw new ROI in blue color

    # Show ROI1
    plt.imshow(img)
    my_roi1.display_roi()
    plt.show(block = False)

    mask = my_roi1.get_mask(img[:, :, 0])
    plt.imshow(mask, interpolation='nearest', cmap='Blues')
    plt.show()

    x1 = np.where(mask == True)  # Get the index as a tuple
    X1 = img[x1[0], x1[1]]  # Put the color information of the pixel with corresponding index into new Array X1, X1---class Blue

    return X1

t1 = cv.getTickCount()  #Start time

result1 = np.zeros((1,3))  #initialization  1X3
c1 = [2,3,8,14,21,25,30,36,42] #black
c = [1,7,10,12,13,16,17,18,19,22,26,27,28,32,33,35,38,39,43] #not barrel blue
c2 = [6]
for i in c2:
    path = "/Users/zenghailong/Desktop/2019Winter/ECE276A/ECE276A_HW1/trainset/{}.png".format(i)
    print(path)
    img = cv.imread(path)
    result = Hand_labeling(img)
    result1 = np.append(result1, result, axis=0)   #First class, class blue
Trainset1 = result1[1:]     #Trainset for class 1, class blue
np.savetxt('Trainset1_RGB_1.txt', Trainset1)

