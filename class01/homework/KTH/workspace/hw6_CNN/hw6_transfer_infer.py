import tensorflow as tf
# Helper libraires
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
img_height = 255
img_width = 255
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE # 병렬연산을 할 것인지에 대한 인자를 알아서 처리하도록.
# Dataset 준비  (https://www.tensorflow.org/tutorials/load_data/images?hl=ko)
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

num = 20
def prepare(ds, batch = 1, shuffle=False, augment=False):
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    # Resize and rescale all datasets.
    # x: image, y: label
    # 이미지 크기 조정
    ds = ds.map(lambda x, y: (tf.image.resize(x, [img_height, img_width]), y),
        num_parallel_calls=AUTOTUNE)
    # Batch all datasets
    ds = ds.batch(batch_size)
    # 데이터 로딩과 모델 학습이 병렬로 처리되기 위해
    # prefetch()를 사용해서 현재 배치가 처리되는 동안 다음 배치의 데이터를 미리 로드하도록 함.
    return ds.prefetch(buffer_size=AUTOTUNE)

num_classes = metadata.features['label'].num_classes
label_name = metadata.features['label'].names
print(label_name, ", classnum : ", num_classes, ", type: ", type(label_name))
test_ds = prepare(test_ds, num)
image_test, label_test = next(iter(test_ds))
image_test = np.array(image_test)
label_test = np.array(label_test, dtype='int')
# 모델 불러오기
model = tf.keras.models.load_model('transfer_learning_flower.keras')
model.summary()
predict = model.predict(image_test)
predicted_classes = np.argmax(predict, axis=1)

print("실제 레이블 | 예측 레이블")
print("------------------------")
for ll in range((label_test.size)):
    print(label_name[label_test[ll]], "|",
label_name[predicted_classes[ll]])
print("------------------------")
# print("실제 레이블:", [label_name[idx] for idx in label_test])
# print("예측 레이블:", [label_name[idx] for idx in predicted_classes])
accuracy = np.mean(predicted_classes == label_test)
print(f"정확도: {accuracy:.2%}")