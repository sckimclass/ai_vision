import tensorflow as tf
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np


from keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale = 1./255)
validation = ImageDataGenerator(rescale = 1./255)

train_dataset = train.flow_from_directory('dataset/train/',
                                                 target_size = (200, 200),
                                                 batch_size = 3,
                                                 class_mode = 'binary')

validataion_dataset = validation.flow_from_directory('dataset/test/',
                                            target_size = (200, 200),
                                            batch_size = 3,
                                             class_mode = 'binary')


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (200, 200, 3)),
                                    tf.keras.layers.MaxPool2D(2,2),

                                    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),

                                    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),

                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(1, activation = 'sigmoid')
                                    ])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(train_dataset,
          steps_per_epoch = 3,
          epochs=20)

dir_path = "dataset/test"

img = image.load_img('/content/dataset/test/dogs/4803.jpg',target_size=(200,200))
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
