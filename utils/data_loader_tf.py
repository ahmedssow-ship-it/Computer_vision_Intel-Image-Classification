
import tensorflow as tf

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

def get_datasets(data_dir):

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Normalisation
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

    return train_dataset, test_dataset
