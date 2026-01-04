# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 17:17:55 2026

@author: V1M

CNN - Object Classification
"""
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten , Conv2D, MaxPool2D
import matplotlib.pyplot as plt

width = 128
num_classes = 2
trainpath = 'train/'
testpath = 'test/'
trainImg = [trainpath+f for f in listdir(trainpath)]
testImg = [testpath+f for f in listdir(testpath)]


def img2data(paths):
    rawImgs = []
    labels = []

    for imagePath in paths:
        label_name = os.path.normpath(imagePath).split(os.sep)[-1]
        label_name = label_name.strip().lower()

        print("Reading class:", label_name)

        for item in tqdm(listdir(imagePath)):
            file = join(imagePath, item)

            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(file)
                img = cv2.resize(img, (width, width))
                rawImgs.append(img)

                if label_name == 'lego':
                    labels.append([1,0])
                elif label_name == 'toybike':
                    labels.append([0,1])
                else:
                    print("Unknown label folder:", label_name)

    return rawImgs, labels

x_train, y_train = img2data(trainImg)
x_test, y_test = img2data(testImg)

# print("Train:", len(x_train), len(y_train))
# print("Test :", len(x_test), len(y_test))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train.shape,y_train.shape,x_test.shape, y_test.shape

model0 = Sequential([

        Conv2D(64, (3,3), activation='relu', input_shape=(width, width, 3)),
        MaxPool2D(pool_size=(2,2 )),
        Conv2D(64,(3,3)),
        MaxPool2D(pool_size=(2,2 )),
        Dropout(0.25),
        Dense(16),
        Flatten(),

        Dense(num_classes, activation='softmax') #softmax for one hot . . # sigmoid for 0/1
    ])

model0.summary()

model0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics= ['accuracy'])

batch_size = 32
epochs = 100

history = model0.fit(x_train, y_train ,batch_size=batch_size, epochs=epochs ,validation_data=(x_test, y_test))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()

# test
testImg = [testpath+f for f in listdir(testpath)]

for imagePath in testImg:
    for item in listdir(imagePath):

        file = join(imagePath, item)
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):

            imgori = cv2.imread(file)
            imgori = cv2.cvtColor(imgori, cv2.COLOR_BGR2RGB)

            img = cv2.resize(imgori, (width, width))
            rimg = img.astype('float32') / 255.0
            rimg = np.expand_dims(rimg, axis=0)

            predict = model0.predict(rimg)
            labels = ['lego', 'Toybike']
            result = labels[np.argmax(predict)]


            cv2.putText(
                imgori,
                f"Predict: {result}",
                (500, 500),
                cv2.FONT_HERSHEY_SIMPLEX,
                10,
                (255, 0, 0),  
                2,
                cv2.LINE_AA
            )

            print(file)
            print("prob:", predict)
            print("predict:", result)

            plt.imshow(imgori)
            plt.axis("off")
            plt.show()