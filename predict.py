import tensorflow as tf
from tensorflow import keras
import image_settings
import matplotlib.pyplot as plt
import math

image_height = image_settings.image_height
image_width = image_settings.image_width
color_mode=image_settings.color_mode

model_dir = "model"

model = keras.models.load_model(model_dir)

prediction_dir = "prediction"

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
    
rows = math.ceil(number_predict_imgs / 3)

plt.figure(figsize=(10,10))
for images, labels in myimagedataset:
	for i in range(number_predict_imgs):
		ax = plt.subplot(3, rows, i + 1)
		if color_mode == "grayscale":
			plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
		else:
			plt.imshow(images[i].numpy().astype("uint8"))
		plt.title("Actual class = " + prediction_data_class_names[labels[i]] + "\nPredicted class = " + all_class_names[predictions[i].argmax()])
		plt.axis("off")
plt.show()

print("fin")