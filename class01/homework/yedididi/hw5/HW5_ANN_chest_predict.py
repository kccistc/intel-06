import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 모델 불러오기
model = tf.keras.models.load_model('medical_ann.h5')

# 이미지 전처리용 ImageDataGenerator (테스트용이므로 augmentation X)
test_datagen = ImageDataGenerator(rescale=1./255)

# 테스트셋 로드
test_set = test_datagen.flow_from_directory(
    './chest_xray/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # 예측과 실제를 매칭하기 위해 섞지 않음
)

# 예측 수행
predictions = model.predict(test_set)

# 결과 이진화 (sigmoid이므로 0.5 기준)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# 실제 정답 라벨
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())

# 정확도 출력
accuracy = np.mean(predicted_classes == true_classes)
print(f'Prediction Accuracy: {accuracy * 100:.2f}%')

# 예시 결과 시각화
import random
from PIL import Image
import os

fig, axs = plt.subplots(5, 5, figsize=(15, 15))  # 5x5로 변경
fig.suptitle('Prediction Samples (5x5)', fontsize=20)

for ax in axs.ravel():
    i = random.randint(0, len(test_set.filenames) - 1)
    img_path = os.path.join('./chest_xray/test', test_set.filenames[i])
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(f"Pred: {class_labels[predicted_classes[i]]}\nTrue: {class_labels[true_classes[i]]}", fontsize=10)
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('predict.png', dpi=300)
plt.show(block=False)

