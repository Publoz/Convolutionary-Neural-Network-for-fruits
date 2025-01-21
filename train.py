#imported from ipynb

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


SIZE = 300 #image size


#load images into right set
#(Was done on google colab so we have the directories for google drive folders containing data)
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True, brightness_range=[0.3, 1.4],
                                   channel_shift_range=0.05, rotation_range=45) #Augmenting

training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/COMP309Project/traindata',
                                                 target_size = (SIZE, SIZE),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/COMP309Project/testdata',
                                               target_size = (SIZE, SIZE),
                                                 batch_size = 1,
                                                 class_mode = 'categorical', shuffle=False)

val_datagen = ImageDataGenerator(rescale = 1./255)
val_set = test_datagen.flow_from_directory('/content/drive/MyDrive/COMP309Project/valdata',
                                               target_size = (SIZE, SIZE),
                                                 batch_size = 1,
                                                 class_mode = 'categorical', shuffle=False)


#Setup model structure
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[SIZE, SIZE, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) #layer 1
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) #layer 2
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) #layer 3
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) #Fully connected
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax')) #Output

cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


cnn.fit(x = training_set, validation_data = val_set, epochs = 30) #train


test_set.reset()

scores = cnn.predict(test_set)
print(scores.shape)
actual = test_set.classes
correct = 0

c = 0
s = 0
t = 0
#Get the amount correct and classses for test set
for i in range (450): 
  score = np.argmax(scores[i]) #Np.argmax because we using softmax - we want index of max
  if(score == actual[i]):
    correct += 1
    if(score == 0):
      c += 1
    elif(score==1):
      s+=1
    else:
      t+=1
print(correct)
print(c)
print(s)
print(t)

val_set.reset()

scores = cnn.predict(val_set)
print(scores.shape)
actual = val_set.classes
correct = 0

c = 0
s = 0
t = 0
#Do same for validation
for i in range (450):
  score = np.argmax(scores[i])
  if(score == actual[i]):
    correct += 1
    if(score == 0):
      c += 1
    elif(score==1):
      s+=1
    else:
      t+=1
print(correct)
print(c)
print(s)
print(t)


cnn.save("FinishedModel")

