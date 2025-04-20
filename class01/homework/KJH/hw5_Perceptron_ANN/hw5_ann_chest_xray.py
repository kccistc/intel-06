import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from PIL import Image # for reading images
# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D,
    Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix # <- define evaluation metrics

mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder= './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'
# train
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
#Normal pic
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
norm_pic_address = train_n+norm_pic
#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)
#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')
a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()

# let's build the CNN model
# Homework #3
# cnn = Sequential()
# #Convolution
# model_in = Input(shape = (64, 64, 3))
# model = Flatten()(model_in)
# # Fully Connected Layers
# model = Dense(activation = 'relu', units = 128) (model)
# model = Dense(activation = 'sigmoid', units = 1)(model)
# CNN 구조로 수정된 모델
input_layer = Input(shape=(64, 64, 3))

# CNN Layer 1
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)  # 64→62→31

# CNN Layer 2
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)  # 31→29→14

# Flatten & Dense
x = Flatten()(x)                       # 14×14×32 = 6272
x = Dense(128, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x)
# Compile the Neural network
model_fin = Model(inputs=input_layer, outputs=output_layer)
model_fin.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])

num_of_test_samples = 600
batch_size = 32
# Fitting the CNN to the images
# The function ImageDataGenerator augments your image by iterating through image as 
# your CNN is getting ready to process that image
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255) #Image normalization.
training_set = train_datagen.flow_from_directory('./chest_xray/train', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('./chest_xray/val/', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(
    './chest_xray/test',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')  

model_fin.summary()

# Homework #4
cnn_model = model_fin.fit(training_set, steps_per_epoch = 163, epochs = 10, validation_data = validation_generator, validation_steps = 624)
test_accu = model_fin.evaluate(test_set,steps=624)

model_fin.save('medical_ann.h5')
print('The testing accuracy is :',test_accu[1]*100, '%')
Y_pred = model_fin.predict(test_set, 100)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)

# 테스트셋에서 배치 하나 추출 (이미지 32개 중 앞 25개 사용)
images, labels = next(iter(test_set))  
images = images[:25]
labels = labels[:25].astype(int)

# 예측
pred_probs = model_fin.predict(images)
pred_labels = (pred_probs > 0.5).astype(int).reshape(-1)

# 클래스 이름 매핑
class_names = ['NORMAL', 'PNEUMONIA']

# 시각화
plt.figure(figsize=(12, 12))
for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(images[i])
    true_label = class_names[labels[i]]
    pred_label = class_names[pred_labels[i]]
    
    # 틀린 예측은 빨간색으로 표시
    color = 'black' if true_label == pred_label else 'red'
    ax.set_title(f"Label target: {true_label}\nLabel predict: {pred_label}",
                 fontsize=8, color=color)
    plt.axis("off")

plt.tight_layout()

plt.savefig('chest_xray.jpg') 
plt.show()

# Accuracy Plot
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train set', 'Validation set'], loc='upper left')
plt.grid()
plt.savefig('chest_xray_accuracy.jpg')  
plt.show()
plt.clf()

# Loss Plot
plt.plot(cnn_model.history['loss'])
plt.plot(cnn_model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train set', 'Validation set'], loc='upper left')
plt.grid()
plt.savefig('chest_xray_loss.jpg')  
plt.show()
plt.clf()
