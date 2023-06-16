import tensorflow as tf
from tensorflow import keras
import image_settings
import matplotlib.pyplot as plt

image_height = image_settings.image_height
image_width = image_settings.image_width
color_mode=image_settings.color_mode

model_dir = "model"

model = keras.models.load_model(model_dir)

prediction_dir = "prediction"

myimagedataset = tf.keras.utils.image_dataset_from_directory(
  prediction_dir,
  image_size=(image_width, image_height),
  color_mode=color_mode
)

predictions = model.predict(myimagedataset)
classes = predictions > 0.5

print(predictions)

all_class_names = ['d10', 'd12', 'd20', 'd4', 'd6', 'd8']
prediction_data_class_names = myimagedataset.class_names
number_predict_imgs = len(myimagedataset.file_paths)


for image in predictions:
    print(all_class_names[image.argmax()] + ' | ')

plt.figure(figsize=(10,10))
for images, labels in myimagedataset.take(1):
    for i in range(number_predict_imgs):
        ax = plt.subplot(3, 3, i + 1)
        if color_mode == "grayscale":
          plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
        else:
          plt.imshow(images[i].numpy().astype("uint8"))
        plt.title("Actual class = " + prediction_data_class_names[labels[i]] + "\nPredicted class = " + all_class_names[predictions[i].argmax()])
        plt.axis("off")
plt.show()

print("fin")