import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 모델 로드
model = tf.keras.models.load_model('mnist.keras')

mnist = tf.keras.datasets.mnist
(f_images_train, f_labels_train), (f_images_test, f_labels_test) = mnist.load_data()
f_images_train, f_images_test = f_images_train / 255.0, f_images_test / 255.0

# 데이터 재구성
f_images_test = f_images_test.reshape(-1, 28, 28, 1)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

model.summary()

num = 10
predict = model.predict(f_images_test[:num])
print("Actual labels:", f_labels_test[:num])
print("Predictions:", np.argmax(predict, axis=1))