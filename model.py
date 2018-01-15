
# coding: utf-8

# In[1]:


#Import dependencies
import os
import csv
import numpy as np
import cv2
import keras
import sklearn


# In[2]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, ELU
from keras.layers.convolutional import Cropping2D , Conv2D



# In[3]:


#Check Keras version
print ("Using Keras version:",keras.__version__)

#File import and data preprocessing
lines = []
with open('E:/Self driving training/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)
        
#Split data for validation and test 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


#Generator function declaration (I LOVE this python feature!)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = 'E:/Self driving training/IMG/' + batch_sample[i].split('\\')[-1]
                    image = cv2.imread(name)
                    images.append(image)
                correction = 0.2
                angle = float(batch_sample[3])
                # angle center image
                angles.append(angle)
                # angle left image
                angles.append(angle + correction)
                # angle right image
                angles.append(angle - correction)

            yield shuffle(np.array(images), np.array(angles))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)






# In[4]:


#Check generator output shape
spitted_data = generator(lines,32)
print(next(spitted_data))


# In[5]:


#create model


def nvidiaSDC_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = nvidiaSDC_model()

#Easier writeup :) 
model.summary()


# In[9]:


#Training model

model.compile(loss='mse', optimizer='adam')

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)




history_object = model.fit_generator(train_generator,
                    epochs=3,
                    steps_per_epoch=8,
                    validation_data=validation_generator,
                    validation_steps=5)
model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])


# In[10]:



