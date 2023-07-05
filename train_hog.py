import tensorflow as tf
from tensorflow import keras
import image_settings
import datetime
import hog

def load_datasets(data):
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

    hog_vectors_train = data[0]
    labels_train = data[1]
    hog_vectors_val = data[2]
    labels_val = data[3]
    
    # Load training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((hog_vectors_train, labels_train))
    train_ds = train_ds.batch(batch_size)
    
    # Load validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices((hog_vectors_val, labels_val))
    val_ds = val_ds.batch(batch_size)
    
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
    data = hog.load_hog_data_as_np_array('hog_data', 'hog_numpy_data.npz')
    train_ds, val_ds = load_datasets(data)

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