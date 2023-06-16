import tensorflow as tf
from tensorflow import keras
import pathlib

batch_size = 8
image_height = 248
image_width = 248

# Acquire training data set. I realise pathlib is overkill here.
data_dir = pathlib.Path("dice_data")

train_ds = keras.utils.image_dataset_from_directory(
    data_dir / pathlib.Path("training"),
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

cl = train_ds.class_names

# Acquire validation data set

val_ds = keras.utils.image_dataset_from_directory(
  data_dir / pathlib.Path("validation"),
  seed=123,
  image_size=(image_height, image_width),
  batch_size=batch_size,
  color_mode="grayscale"
)

# Tune data (don't know what this does)
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 6

model = keras.Sequential([
  keras.layers.Rescaling(1./255),
  keras.layers.Conv2D(128, 3, activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(128, 3, activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(128, 3, activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(num_classes),
  keras.layers.Activation('softmax')
])

model.compile(
  optimizer='adam',
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'],
)

with tf.device("/gpu:0"):
  model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
  )

save_dir = pathlib.Path("/mnt/c/Users/Greg/Desktop/Dice_Recognition/model")
model.save(save_dir)

print('fin')