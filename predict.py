import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

model_dir = "model"

model = keras.models.load_model(model_dir)

prediction_dir = "prediction"

myimagedataset = tf.keras.utils.image_dataset_from_directory(
  prediction_dir,
  image_size=(248, 248),
  color_mode='grayscale'
)

predictions = model.predict(myimagedataset)
classes = predictions > 0.5

print(predictions)

class_names = ['d10', 'd12', 'd20', 'd4', 'd6', 'd8']


for image in predictions:
    print(class_names[image.argmax()] + ' | ')

# for i in classes:


# plt.figure(figsize=(10,10))
# for images, labels in myimagedataset.take(1):
#     for i in range(4):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()

print("fin")