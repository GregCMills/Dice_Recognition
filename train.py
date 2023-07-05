import tensorflow as tf
from tensorflow import keras
import image_settings
import datetime

def load_datasets(data_dir):
	"""
	This function loads the training and validation datasets from the specified directory.

	Args:
		data_dir (str): The directory containing the training and validation datasets.

	Returns:
		train_ds (tf.data.Dataset): The training dataset.
		val_ds (tf.data.Dataset): The validation dataset.
	"""
	
	# Retrieve batch size, image height, image width, and color mode from image_settings
	batch_size = image_settings.batch_size
	image_height = image_settings.image_height
	image_width = image_settings.image_width
	color_mode=image_settings.color_mode

	# Load training dataset from specified directory
	train_ds = keras.utils.image_dataset_from_directory(
		data_dir + "/training",
		seed=123,
		image_size=(image_height, image_width),
		batch_size=batch_size,
		color_mode=color_mode
	)
	
	# Load validation dataset from specified directory
	val_ds = keras.utils.image_dataset_from_directory(
		data_dir + "/validation",
		seed=123,
		image_size=(image_height, image_width),
		batch_size=batch_size,
		color_mode=color_mode
	)
	
	# Cache the datasets to improve performance
	AUTOTUNE = tf.data.AUTOTUNE
	train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
	
	return train_ds, val_ds

def create_model():
	"""
	This function creates a convolutional neural network (CNN) model for image classification.

	Returns:
		model (keras.Sequential): The CNN model.
	"""
	num_classes = image_settings.num_classes

	model = keras.Sequential([
		keras.layers.Rescaling(1./255),
		keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
		keras.layers.MaxPooling2D(strides=2),
		keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
		keras.layers.MaxPooling2D(strides=2),
		keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
		keras.layers.MaxPooling2D(strides=2),
		keras.layers.Flatten(),
		keras.layers.Dropout(0.5),  # Add dropout layer with a dropout rate of 0.5
		keras.layers.Dense(num_classes),
		keras.layers.Activation('softmax')
	])
	
	model.compile(
		optimizer='adam',
		loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
		metrics=[keras.metrics.SparseCategoricalAccuracy()],
	)
	
	return model

def main():
	# Load the dataset
	data_dir = "dice_data"
	train_ds, val_ds = load_datasets(data_dir)
	
	model = create_model()
	
	# Set the directory path for TensorBoard logs
	log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	# Create the TensorBoard callback
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	
	# Train the model
	with tf.device("/gpu:0"):
		model.fit(
			train_ds,
			validation_data=val_ds,
			epochs=5,
			callbacks=[tensorboard_callback]
		)
	
	# Save the model
	save_dir = "model"
	model.save(save_dir)
	
	print('fin')

if __name__ == '__main__':
	main()