import tensorflow as tf
import tensorflow_datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

img_height = 255
img_width = 255
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

(train_ds, val_ds, test_ds), metadata = tensorflow_datasets.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    as_supervised=True,
    with_info=True,
)
num = 20


def prepare(ds, batch=1, shuffle=False, augment=False):
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    ds = ds.map(lambda x, y: (tf.image.resize(x, (img_height, img_width)), y),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds.prefetch(buffer_size=AUTOTUNE)


num_classes = metadata.features['label'].num_classes
label_names = metadata.features['label'].names
print(label_names, ", classes: ", num_classes, ", type: ", type(label_names))

test_ds = prepare(test_ds, num)
image_test, label_test = next(iter(test_ds))
image_test = np.array(image_test)
label_test = np.array(label_test, dtype=int)

model = tf.keras.models.load_model('tf_flowers_model.keras')
model.summary()

predict = model.predict(image_test)
predicted_class = np.argmax(predict, axis=1)

print(" *    Data:   ", label_test)
print(" * Predicted: ", predicted_class)
accuracy = np.mean(label_test == predicted_class)
print("Accuracy: ", accuracy)
