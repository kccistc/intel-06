import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(f_images_train, f_labels_train), (f_images_test, f_labels_test) = mnist.load_data()
f_imagetrain, f_images_test = f_images_train / 255.0, f_images_test / 255.0
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_images_train[i])
    plt.xlabel(class_names[f_labels_train[i]])
plt.show()





# ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 

model.fit(f_images_train, f_labels_train, epochs=10, batch_size=10)
model.summary()
model.save('fashion_mnist.h5')