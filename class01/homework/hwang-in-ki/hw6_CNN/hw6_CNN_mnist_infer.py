import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')

model = tf.keras.models.load_model('fashion_mnist_model.keras')

num = 10
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test,
                                 f_label_test) = fashion_mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0
predict = model.predict(f_image_test[:num])
print(" *    Data:   ", f_label_test[:num])
print(" * Predicted: ", np.argmax(predict, axis=1))
plt.figure(figsize=(10, 10))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for i in range(num):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_test[i], cmap=plt.cm.binary)
    plt.xlabel('   Data:   ' + class_names[f_label_test[i]] + '\n' +
               'Predicted: ' + class_names[np.argmax(predict[i])])
plt.show()
