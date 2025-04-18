import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.models.load_model('fashion_mnist.keras')
mnist = tf.keras.datasets.mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
f_image_train, fimage_test = f_image_train/255, f_image_test /255.0
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

model.summary()

num = 10
predict = model.predict(f_image_test[:num])
print(f_label_test[:num])
print(" * Actual idx, ", f_label_test[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))