#importing required libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import os

import matplotlib.pyplot as plt

dir = r"C:\Users\Greg\Desktop\Dice_Recognition\dice_data\training_hog\d10"

for filename in os.listdir(dir):
    f = os.path.join(dir, filename)
    img = imread(f)
    resized_img = resize(img, (480, 480))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 20))
    #plt.imshow(hog_image_rescaled, cmap = plt.cm.gray)
    #plt.show()
    plt.imsave(f, hog_image_rescaled, cmap = plt.cm.gray)
    print("saved: " + f)