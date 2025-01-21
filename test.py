
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


SIZE = 300

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('testdata',
                                               target_size = (SIZE, SIZE),
                                                 batch_size = 1,
                                                 class_mode = 'categorical', shuffle=False)

cnn = keras.models.load_model('model')

test_set.reset()

scores = cnn.predict(test_set)
print("Predictions done")
actual = test_set.classes
correct = 0

c = 0
s = 0
t = 0
for i in range (len(scores)):
  score = np.argmax(scores[i])
  if(score == actual[i]):
    correct += 1
    if(score == 0):
      c += 1
    elif(score==1):
      s+=1
    else:
      t+=1
print("Total correct")
print(correct)
print("Cherry: ")
print(c)
print("Strawberry: ")
print(s)
print("Tomato: ")
print(t)