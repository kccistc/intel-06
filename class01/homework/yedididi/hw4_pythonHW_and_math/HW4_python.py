# problem no.1
t1 = ('A', 'B')
t1 = t1 + ('C',)

for v in t1:
    print(v)


# problem no.2
x = input()
x = int(x)

for i in range(1, x + 1, 1):
    for j in range(1, x + 1, 1):
        print((i - 1) * x + j, end = " ")
    print("")


# problem no.3
import numpy as np

x = input()
x = int(x)

array = [[0 for col in range(x)] for row in range(x)]

for i in range(1, x + 1, 1):
    for j in range(1, x + 1, 1):
        array[i - 1][j - 1] = ((i - 1) * x) + j

a = np.reshape(array, (1, -1))

print(a)


# problem no.4
import numpy as np
import cv2
from PIL import Image

image = cv2.imread("piplup.jpeg", cv2.IMREAD_COLOR)

image_reshaped = np.expand_dims(image, axis=0)

# print(image_reshaped.shape)

image_changed = np.transpose(image_reshaped, (0, 3, 2, 1))

print(image_changed.shape)
