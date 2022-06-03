# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:25:35 2022

@author: Corne
"""

import numpy as np
from tensorflow import keras
#import keras
import seaborn as sns
from matplotlib import pyplot as plt
#from scipy.io import loadmat
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(20)

# Load the data



# Data augmentation

datagen = ImageDataGenerator(validation_split=0.1)



train_generator=datagen.flow_from_dataframe(
    dataframe=train_df, directory="temp folder direc", 
    x_col="image_path", y_col="target", has_ext=True, seed = 20,
    class_mode="raw", target_size=(640,640), batch_size=128, subset = "training")

validation_generator = datagen.flow_from_dataframe(dataframe=train_df, directory="temp folder direc", 
    x_col="image_path", y_col="target", has_ext=True, seed = 20,
    class_mode="raw", target_size=(640,640), batch_size=128, subset = "validation")


test_generator = test_datagen.flow_from_directory(directory="temp folder direc", 
    seed = 20, class_mode="raw", target_size=(640,640), batch_size=1, shuffle = False)



# Define model

keras.backend.clear_session()

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3),
                        activation='relu',
                        input_shape=(640, 640, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3),
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(64, (3, 3),
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3),
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(128, (3, 3),
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3),
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(1)
])

early_stopping = keras.callbacks.EarlyStopping(patience=8,
                                               restore_best_weights=True,
                                               min_delta=.5)

optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'model_cnn.h5',
    save_best_only=True,
    save_freq="epoch")

model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['mae','mse'])

model.summary()

# Fit model in order to make predictions

history = model.fit_generator(generator=train_generator,
                                     steps_per_epoch=128,
                                     validation_data=validation_generator,
                                     validation_steps=128,
                                     epochs=100,
                                     callbacks=[early_stopping,model_checkpoint])

# Evaluate model on test data
test_generator.reset()
pred = model.evaluate(test_generator,callbacks=callbacks)

print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.
      format(test_acc, test_loss))

loss=history.history['loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,label='Training loss')
plt.legend()
plt.show()

# Get predictions and apply inverse transformation to the labels

# Plot the confusion matrix

matrix = confusion_matrix(y_train, y_pred, labels=lb.classes_)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(matrix, annot=True, cmap='Greens', fmt='d', ax=ax)
plt.title('Confusion Matrix for training dataset')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
