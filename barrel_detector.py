'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np

class BarrelDetector():
	def __init__(self):
		'''
			Initilize your blue barrel detector with the attributes you need
			eg. parameters of your classifier
		'''
		w = np.array([[-23.61425758],[-10.3807786],[17.7791692]])
		self.w = w

	def softmax(self, z):
		return np.exp(z) / np.sum(np.exp(z))

	def segment_image(self, img):
		'''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		# YOUR CODE HERE
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		m, n = img.shape[0:2]
		mask_img = np.zeros((m, n))
		for i in range(m):
			for j in range(n):
				p = 1.0/(1+np.exp(-(np.dot(img[i, j, :]/255, self.w))))
				if p > 0.7:
					mask_img[i, j] = 1
				else:
					mask_img[i, j] = 0

		return mask_img

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		# YOUR CODE HERE
		mask_img = self.segment_image(img)
		mask_img = cv2.convertScaleAbs(mask_img)  # converted into uint8


		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # Create a structing element
		mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)  # Erosion followed by dilation

		label_img = label(mask_img, connectivity=1)
		props = regionprops(label_img)
		boxes = []
		area = []
		for prop in props:
			area.append(prop.area)

		for prop in props:
			xmin, ymin, xmax, ymax = prop.bbox
			y = ymax - ymin
			x = xmax - xmin
			if prop.area / max(area) > 0.6:
				boxes.append([ymin, xmin, ymax, xmax])

			elif prop.area / max(area) > 0.2:
				if min(x, y) / max(x, y) > 0.5 and min(x, y) / max(x, y) < 0.6:
					boxes.append([ymin, xmin, ymax, xmax])
		return boxes

if __name__ == '__main__':
	folder = "trainset"
	my_detector = BarrelDetector()
	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope

