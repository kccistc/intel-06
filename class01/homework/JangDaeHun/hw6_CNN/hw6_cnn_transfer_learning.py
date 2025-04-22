import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Conv2D
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pickle

img_height = 255
img_width = 255
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE 

(train_ds, val_ds, test_ds), metadata = tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                  with_info=True, as_supervised=True, )
num_classes = metadata.features['label'].num_classes
label_name = metadata.features['label'].names
print(label_name, ", classnum : ", num_classes)

def prepare(ds, shuffle=False, augment=False):
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    ds = ds.map(lambda x, y: (tf.image.resize(x, [img_height, img_width]), y), num_parallel_calls=AUTOTUNE)
    
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    
    ds = ds.batch(batch_size)
    if augment:
        data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

base_model = tf.keras.applications.MobileNetV3Small(
    weights='imagenet', input_shape = (img_height, img_width, 3), include_top = False)
base_model.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, 3))

x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes,activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_ds, epochs=15, validation_data=val_ds)
model.save('transfer_learning_flower.keras')
with open('history_flower', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)