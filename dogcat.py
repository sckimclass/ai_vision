import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 훈련 셋, 검증 셋 저장위치 지정
train_dir = 'dataset/train/'
valid_dir = 'dataset/test/'

# 이미지 데이터 제너레이터 정의 (Augmentation 적용)
image_gen = ImageDataGenerator(rescale=1/255., 
                                   horizontal_flip=True,
                                   rotation_range=35,                                
                                   zoom_range=0.2)

# flow_from_directory 함수로 폴더에서 이미지 가져와서 제너레이터 객체로 정리 
train_gen = image_gen.flow_from_directory(train_dir, 
                                                  batch_size=32, 
                                                  target_size=(224,224),   
                                                  classes=['cats','dogs'], 
                                                  class_mode = 'binary', 
                                                  seed=2020)

valid_gen = image_gen.flow_from_directory(valid_dir,  
                                                  batch_size=32, 
                                                  target_size=(224,224),   
                                                  classes=['cats','dogs'], 
                                                  class_mode = 'binary', 
                                                  seed=2020)

model = tf.keras.models.Sequential([
        # Convolution 층 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Classifier 출력층 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'), 
        tf.keras.layers.Dropout(0.3),              
        tf.keras.layers.Dense(1, activation='sigmoid'),
])

# 모델 컴파일
model.compile(optimizer=tf.optimizers.Adam(lr=0.001),  
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# 모델 훈련
history = model.fit(train_gen, validation_data=valid_gen, epochs=40)

# 손실함수, 정확도 그래프 그리기 
loss, val_loss = history.history['loss'], history.history['val_loss']
acc, val_acc = history.history['accuracy'], history.history['val_accuracy']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, 40 + 1), loss, label='Training')
axes[0].plot(range(1, 40 + 1), val_loss, label='Validation')
axes[0].legend(loc='best')
axes[0].set_title('Loss')

axes[1].plot(range(1, 40 + 1), acc, label='Training')
axes[1].plot(range(1, 40 + 1), val_acc, label='Validation')
axes[1].legend(loc='best')
axes[1].set_title('Accuracy')

plt.show()

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

# 예측
dir_path = "dataset/test"

img = image.load_img('/content/dataset/test/cats/4829.jpg',target_size=(224,224))
#img = image.load_img('/content/sckim.jpg',target_size=(224,224))
plt.imshow(img)
plt.show()

X = image.img_to_array(img)
X = np.expand_dims(X, axis = 0)
images = np.vstack([X])
val = model.predict(images)
if val == 0:
  print("cats")
else:
  print("dogs")
