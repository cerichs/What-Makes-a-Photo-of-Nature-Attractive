# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:25:35 2022

@author: Corne
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
#import keras
import seaborn as sns
from matplotlib import pyplot as plt
#from scipy.io import loadmat
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image


np.random.seed(20)

# Load the data

train_df=pd.read_pickle("train_df.pkl")
test_df=pd.read_pickle("test_df.pkl")

# Data augmentation

datagen = ImageDataGenerator(rescale=1./255,validation_split=0.1)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator=datagen.flow_from_dataframe(
    dataframe=train_df, directory="../images", 
    x_col="path", y_col="residfaves", seed = 20,
    class_mode="raw", color_mode='rgb', target_size=(640,640), batch_size=32, subset = "training")

validation_generator = datagen.flow_from_dataframe(dataframe=train_df, directory="../images", 
    x_col="path", y_col="residfaves", seed = 20,
    class_mode="raw",color_mode='rgb', target_size=(640,640), batch_size=32, subset = "validation")


test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, directory="../images",
    x_col="path", y_col="residfaves",seed = 20, class_mode="raw", target_size=(640,640), 
    batch_size=1, shuffle = False)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



# Define model

keras.backend.clear_session()

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3),
                        activation='relu',
                        input_shape=(640, 640, 3)),
    
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(64, (3, 3),
                        activation='relu'),
    
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(64, (3, 3),
                        activation='relu'),
    
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(64, (3, 3),
                        activation='relu'),
    
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(128, (3, 3),
                        activation='relu'),
    
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(256, (3, 3),
                        activation='relu'),
    
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(256, (3, 3),
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(512, (3, 3),
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    #keras.layers.Flatten(),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(1)
])

early_stopping = keras.callbacks.EarlyStopping(patience=15, monitor="val_loss",
                                               restore_best_weights=True,verbose=2)

optimizer = keras.optimizers.Adam(lr=0.001, amsgrad=True)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'model_cnn.h5',
    monitor="val_loss",
    verbose=2,
    save_best_only=True,
    save_freq="epoch")

model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mae','mse'])

model.summary()

# Fit model in order to make predictions

history = model.fit_generator(generator=train_generator,
                                     steps_per_epoch=128,
                                     validation_data=validation_generator,
                                     validation_steps=128,
                                     epochs=50,
                                     callbacks=[early_stopping,model_checkpoint])
#model.save_weights("model.h5")

# Evaluate model on test data
test_generator.reset()
pred = model.evaluate(test_generator)

#print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.
      #format(test_acc, test_loss))

loss=history.history['loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,label='Training loss')
plt.legend()
plt.show()

# Get predictions and apply inverse transformation to the labels

# Plot the confusion matrix

#matrix = confusion_matrix(y_train, y_pred, labels=lb.classes_)

#fig, ax = plt.subplots(figsize=(14, 12))
#sns.heatmap(matrix, annot=True, cmap='Greens', fmt='d', ax=ax)
#plt.title('Confusion Matrix for training dataset')
#plt.xlabel('Predicted label')
#plt.ylabel('True label')
#plt.show()
