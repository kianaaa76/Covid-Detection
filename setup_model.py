import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,ReLU, Dropout
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import accuracy_score, classification_report
import os
import cv2
from tensorflow import keras
import tensorflow as tf
from keras.models import Model


opt = RMSprop(
    learning_rate=0.0001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop")
labels = ['0', '1']
img_size = 224
num_classes = 2

def get_data(data_dir):
    data = [] 
    for label in labels: 
        if (label == '0'):
          path = os.path.join(data_dir, 'negative')
        else:
          path = os.path.join(data_dir, 'positive')
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, int(label)])
            except Exception as e:
                print(e)
    return(data)

train = get_data('/content/drive/MyDrive/image_project_train')
test = get_data('/content/drive/MyDrive/image_project_test')
valid = get_data('/content/drive/MyDrive/image_project_validation')

x_train = []
y_train = []
x_test = []
y_test = []
x_valid = []
y_valid = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in test:
  x_test.append(feature)
  y_test.append(label)

for feature, label in valid:
  x_valid.append(feature)
  y_valid.append(label)

X = []
Y = []

for feature, label in train:
  X.append(feature)
  Y.append(label)

for feature, label in test:
  X.append(feature)
  Y.append(label)

for feature, label in valid:
  X.append(feature)
  Y.append(label)

# Normalize the data
x_train = np.array(x_train) / img_size
x_test = np.array(x_test) / img_size
x_valid = np.array(x_valid) / img_size
X = np.array(X) / img_size

x_train = np.expand_dims(x_train, axis=-1)
y_train = np.array(y_train)

x_test = np.expand_dims(x_test, axis=3)
y_test = np.array(y_test)

x_valid = np.expand_dims(x_valid, axis=3)
y_valid = np.array(y_valid)

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

X = np.expand_dims(X, axis=3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224,224,1), activation='relu',strides=1,padding='valid'))
model.add(Conv2D(128, (3, 3), input_shape=(222, 222, 32), activation='relu',strides=1,padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), input_shape=(110, 110, 128), activation='relu',strides=1,padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), input_shape=(54, 54, 64), activation='relu',strides=1,padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(512, (3, 3), input_shape=(26, 26, 128), activation='relu',strides=1,padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(512, (3, 3), input_shape=(12, 12, 512), activation='relu',strides=1,padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=20, batch_size=50,validation_data=(x_valid, y_valid))