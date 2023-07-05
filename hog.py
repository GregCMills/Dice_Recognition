#importing required libraries
# from skimage.io import imread
# from skimage.transform import resize
# from skimage.feature import hog
# from skimage import exposure
# import os

# import matplotlib.pyplot as plt

# dir = r"C:\Users\Greg\Desktop\Dice_Recognition\dice_data\training_hog\d10"

# for filename in os.listdir(dir):
#     f = os.path.join(dir, filename)
#     img = imread(f)
#     resized_img = resize(img, (480, 480))
#     fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
#     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 20))
#     #plt.imshow(hog_image_rescaled, cmap = plt.cm.gray)
#     #plt.show()
#     plt.imsave(f, hog_image_rescaled, cmap = plt.cm.gray)
#     print("saved: " + f)

import numpy as np
import cv2
import os
from skimage.feature import hog
from skimage import exposure
import image_settings

# Function to extract HOG features from an image
def extract_hog_features(image):
	hog_features = hog(image, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
	return hog_features

# Function to load and preprocess the image dataset
def load_image_dataset_as_hog(dataset_path):
	image_filenames = os.listdir(dataset_path)
	hogs = []
	labels = []
	width = image_settings.image_width
	height = image_settings.image_height

	for label in image_filenames:
		if label != '.gitignore':
			folder_dir = dataset_path + '/' + label
			label_folder = os.listdir(folder_dir)
			print('starting: ' + label)
			for filename in label_folder:
				# Load and preprocess the image
				image = cv2.imread(os.path.join(folder_dir, filename))
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				image = cv2.resize(image, (width, height))

				# Extract HOG features
				hog_features = extract_hog_features(image)

				# Post Process the image
				hog_features = exposure.rescale_intensity(hog_features, in_range=(0, 20))

				# Store the features and label
				hogs.append(hog_features)
				labels.append(label) # Assuming the label is in the filename

	return np.array(hogs), np.array(labels)

def save_data_as_hog(hog_filepath):
	# Path to the image dataset
	training_dataset_path = 'dice_data/training'
	# Path to the image dataset
	validation_dataset_path = 'dice_data/validation'

	# Load and preprocess the image dataset
	training_hog_vectors, training_labels = load_image_dataset_as_hog(training_dataset_path)
	validation_hog_vectors, validation_labels = load_image_dataset_as_hog(validation_dataset_path)

	np.savez(hog_filepath, training_hog_vectors=training_hog_vectors, training_labels=training_labels, validation_hog_vectors=validation_hog_vectors, validation_labels=validation_labels)

def load_hog_data_as_np_array(hog_dir, hog_filename):
	hog_filepath = hog_dir + '/' + hog_filename
	
	files_in_hog_dir = os.listdir(hog_dir)
	if hog_filename not in files_in_hog_dir:
		save_data_as_hog(hog_filepath)
		
	data = np.load(hog_filepath)
	return data
		
# def main():
# 	hog_dir = 'hog_data'
# 	hog_filename = 'hog_numpy_data.npz'


# if __name__ == '__main__':
# 	main()