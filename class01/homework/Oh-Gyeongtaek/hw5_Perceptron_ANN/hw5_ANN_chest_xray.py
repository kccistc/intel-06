import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image # for reading images

# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix # <- define evaluation metrics
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 비활성화
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
sic_pic =  os.listdir(train_p)[rand_norm]
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

#hw3
model_in = Input(shape = (64, 64, 3))
model = Flatten()(model_in)

model = Dense(activation = 'relu', units = 128) (model)
model = Dense(activation = 'sigmoid', units = 1)(model)

model_fin = Model(inputs=model_in, outputs=model)
model_fin.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

num_of_test_samples = 600
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./chest_xray/train',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('./chest_xray/val/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')
test_set = test_datagen.flow_from_directory('./chest_xray/test', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
model_fin.summary()

#hw4
cnn_model = model_fin.fit(training_set,
steps_per_epoch = 163,
epochs = 10,
validation_data = validation_generator,
validation_steps = 624)
test_accu = model_fin.evaluate(test_set,steps=624)
model_fin.save('medical_ann.h5')
print('The testing accuracy is :',test_accu[1]*100, '%')
Y_pred = model_fin.predict(test_set, 100)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)

#hw5
test_generator = test_datagen.flow_from_directory(
    './chest_xray/test', 
    target_size=(64, 64),
    batch_size=25,
    class_mode='binary',
    shuffle=False
)

test_images, test_labels = next(test_generator)

predictions = model_fin.predict(test_images)
predictions = (predictions > 0.5).astype(int)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_images[i])
    plt.axis('off')
    true_label = "NORMAL" if test_labels[i] == 0 else "PNEUMONIA"
    pred_label = "NORMAL" if predictions[i][0] == 0 else "PNEUMONIA"
    plt.title(f"Label target: {true_label}\nLabel predict: {pred_label}", fontsize=8)

plt.tight_layout()
plt.savefig('chest_xray_predictions.png', dpi=300)
plt.show()

#hw6
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(cnn_model.history['accuracy'], 'b-', label='Train set')
# plt.plot(cnn_model.history['val_accuracy'], 'orange', label='Validation set')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.ylim(0.6, 0.9)
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(cnn_model.history['loss'], 'b-', label='Train set')
# plt.plot(cnn_model.history['val_loss'], 'orange', label='Validation set')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.ylim(0.2, 0.9)
# plt.legend()

# plt.tight_layout()
# plt.savefig('chest_xray_loss.png', dpi=300)
# plt.show()

# Accuracy
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=False)
plt.clf()

# Loss
plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
plt.clf()