import tensorflow as tf
from tensorflow import keras
import image_settings
import matplotlib.pyplot as plt
import math
import gradCAM as gcam
import cv2

image_height = image_settings.image_height
image_width = image_settings.image_width
color_mode=image_settings.color_mode

model_dir = "model"
model = keras.models.load_model(model_dir)

prediction_dir = "pred_from_val"

myimagedataset = tf.keras.utils.image_dataset_from_directory(
  prediction_dir,
  image_size=(image_width, image_height),
  color_mode=color_mode,
  shuffle=False
)

predictions = model.predict(myimagedataset, verbose=2)
print("Pred = " + str(predictions))

eval = model.evaluate(myimagedataset)
print("eval = " + str(eval))

all_class_names = ['d10', 'd12', 'd20', 'd4', 'd6', 'd8']
prediction_data_class_names = myimagedataset.class_names
number_predict_imgs = len(myimagedataset.file_paths)


for image in predictions:
    print(all_class_names[image.argmax()] + ' | ')
    

cols = 7    
rows = math.ceil(number_predict_imgs / cols)

plt.figure(figsize=(10,10))
plt.suptitle(prediction_dir)
# for images, labels in myimagedataset:
# 	for i in range(number_predict_imgs):
# 		ax = plt.subplot(cols, rows, i + 1)
# 		if color_mode == "grayscale":
# 			plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
# 		else:
# 			plt.imshow(images[i].numpy().astype("uint8"))
# 		plt.title("Act: " + prediction_data_class_names[labels[i]] + "\nPre: " + all_class_names[predictions[i].argmax()])
# 		plt.axis("off")
# plt.show()

# Remove last layer's softmax
#model.layers[-1].activation = None

for images, labels in myimagedataset:
	for i in range(number_predict_imgs):
		ax = plt.subplot(cols, rows, i + 1)
		if color_mode == "grayscale":
			plt.imshow(gcam.overlayed_heatmap(images[i], model, 'conv2d_2', alpha=0.7, pred_index=4), cmap="gray")
		else:
			plt.imshow(gcam.overlayed_heatmap(images[i], model, 'conv2d_2', alpha=0.7, pred_index=2))
		plt.title("Act: " + prediction_data_class_names[labels[i]] + "\nPre: " + all_class_names[predictions[i].argmax()])
		plt.axis("off")
plt.show()


print("fin")