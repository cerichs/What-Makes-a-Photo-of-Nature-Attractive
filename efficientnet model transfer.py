# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:31:04 2022

@author: Corne
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from keras.preprocessing.image import img_to_array
import seaborn as sn


np.random.seed(20)

# Load the data

train_df=pd.read_pickle("train_df_class_str.pkl")
test_df=pd.read_pickle("test_df_class_str.pkl")

# Data augmentation

datagen = ImageDataGenerator(validation_split=0.1,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2)
test_datagen = ImageDataGenerator()


train_generator=datagen.flow_from_dataframe(
    dataframe=train_df, directory="../images", 
    x_col="path", y_col="class", seed = 20,
    class_mode="categorical", color_mode='rgb', target_size=(224,224), batch_size=64, subset = "training")

validation_generator = datagen.flow_from_dataframe(dataframe=train_df, directory="../images", 
    x_col="path", y_col="class", seed = 20,
    class_mode="categorical",color_mode='rgb', target_size=(224,224), batch_size=64, subset = "validation")


test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, directory="../images",
    x_col="path", y_col="class",seed = 20, class_mode="categorical", target_size=(224,224), 
    batch_size=1, shuffle = False)


##################################################################
#### Magical code from Tensorflow to help with GPU OOM issues ####
#### Source: https://www.tensorflow.org/guide/gpu             ####
##################################################################


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

keras.backend.clear_session()

ENB= EfficientNetB0(weights='imagenet',include_top=False)
ENB.trainable = False ## Freeze weights
model = keras.Sequential()
model.add(ENB)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(4,activation="softmax"))

model.summary()

early_stopping = keras.callbacks.EarlyStopping(patience=15, monitor="val_accuracy",
                                               restore_best_weights=True,verbose=2)

optimizer = keras.optimizers.Adam(lr=0.001)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'model_cnn_class.h5',
    verbose=2,
    save_best_only=True,
    save_freq="epoch")

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

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

### Fine tune EfficientNet

ENB.trainable = True
model.summary()
optimizer = keras.optimizers.Adam(lr=1e-5)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                                     steps_per_epoch=128,
                                     validation_data=validation_generator,
                                     validation_steps=128,
                                     epochs=10,
                                     callbacks=[early_stopping,model_checkpoint])

test_generator.reset()
pred = model.evaluate(test_generator)

def get_predicts(model,test_df):
    temp,temp_path,temp_true=[],[],[]
    temp_df=test_df.reset_index(drop=True)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    for i in range(len(temp_df)):
    #for i in range(100):
        try:
            img = Image.open(temp_df.iloc[i,2])
            img_input = np.expand_dims(img_to_array(img), axis=0)
            predictions=model.predict(img_input)
            pred_labels = np.argmax(predictions, axis = 1) 
            predictions = [labels[k] for k in pred_labels]
            temp.append(predictions[0])
            temp_path.append(temp_df.iloc[i,2])
            temp_true.append(temp_df.iloc[i,5])
            if i%1000==0:
                print(i)
        except:
            pass
    predict_df=pd.DataFrame()
    predict_df["Image path"]=temp_path
    predict_df["Predicted Class"]=temp
    predict_df["True Class"]=temp_true
    predict_df.to_pickle("predicted_df_enb_classifi.pkl")
    return predict_df

predict_df=get_predicts(model,test_df)

class_names=["not attractive","attractive","medium attractive","very attractive"]
CM = confusion_matrix(predict_df["True Class"],predict_df["Predicted Class"],labels=class_names)
ax = plt.axes()
sn.heatmap(CM, annot=True,cmap="Greens", yticklabels=class_names, xticklabels=class_names, ax = ax,fmt="d")
#sn.heatmap(CM, annot=True,cmap="Greens",ax = ax,fmt="d")
ax.set_ylabel("True Class")
ax.set_xlabel("Predicted Class")
ax.set_title('Confusion matrix')
plt.savefig('confusion_matrix_enb_0.377.png',
            dpi=300,bbox_inches='tight')
plt.show()
