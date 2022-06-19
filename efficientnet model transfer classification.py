# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:31:04 2022

@author: Corne
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from keras.preprocessing.image import img_to_array
#from vis.utils import utils

def get_predicts(model,test_df):
    temp,temp_path,temp_true=[],[],[]
    temp_df=test_df.reset_index(drop=True)
    for i in range(len(temp_df)):
        try:
            img = Image.open(temp_df.iloc[i,2])
            img_input = np.expand_dims(img_to_array(img), axis=0)
            temp.append(model.predict(img_input)[0][0])
            temp_path.append(temp_df.iloc[i,2])
            temp_true.append(temp_df.iloc[i,0])
            if i%1000==0:
                print(i)
        except:
            pass
    predict_df=pd.DataFrame()
    predict_df["Image path"]=temp_path
    predict_df["Predicted Resid"]=temp
    predict_df["True Resid"]=temp_true
    predict_df.to_pickle("predicted_df_enb_standard.pkl")
    return predict_df


np.random.seed(20)

# Load the data

#train_df=pd.read_pickle("train_df.pkl")
#test_df=pd.read_pickle("test_df.pkl")

train_df=pd.read_pickle("train_df.pkl")
test_df=pd.read_pickle("test_df.pkl")

# Data augmentation

datagen = ImageDataGenerator(validation_split=0.1,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2)
test_datagen = ImageDataGenerator()

##### IMAGE GENERATORS

train_generator=datagen.flow_from_dataframe(
    dataframe=train_df, directory="../images", 
    x_col="path", y_col="residfaves", seed = 20,
    class_mode="raw", color_mode='rgb', target_size=(224,224), batch_size=64, subset = "training")

validation_generator = datagen.flow_from_dataframe(dataframe=train_df, directory="../images", 
    x_col="path", y_col="residfaves", seed = 20,
    class_mode="raw",color_mode='rgb', target_size=(224,224), batch_size=64, subset = "validation")


test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, directory="../images",
    x_col="path", y_col="residfaves",seed = 20, class_mode="raw", target_size=(224,224), 
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
    


#### IMAGE TRANSFER LEARNING


keras.backend.clear_session()

ENB= EfficientNetB0(weights='imagenet',include_top=False)
ENB.trainable = False ## Freeze weights
model = keras.Sequential()
model.add(ENB)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1,activation="linear"))

model.summary()

rlronp=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,
                                            patience=1, verbose=1)

early_stopping = keras.callbacks.EarlyStopping(patience=15, monitor="val_loss",
                                               restore_best_weights=True,verbose=2)

optimizer = keras.optimizers.Adam(lr=0.001)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'model_cnn_reg.h5',
    monitor="val_loss",
    verbose=2,
    save_best_only=True,
    save_freq="epoch")

model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mse','mae'])

model.summary()


history = model.fit_generator(generator=train_generator,
                                     steps_per_epoch=256,
                                     validation_data=validation_generator,
                                     validation_steps=128,
                                     epochs=50,
                                     callbacks=[rlronp,early_stopping,model_checkpoint])
#model.save_weights("model.h5")

# Evaluate model on test data
test_generator.reset()
pred = model.evaluate(test_generator)

#model.save('EfficientNetB0_reg_trans_0.5669.h5')


predict_df=get_predicts(model,test_df)

plt.scatter(predict_df["Predicted Resid"],predict_df["True Resid"],alpha=0.5,s=5)
plt.plot([-5, 2], [-5 , 2], 'k-', color = 'r')
plt.xlabel("Predicted Residfaves")
plt.ylabel("True Residfaves")
plt.title(f"Scatterplot of Predicted Residfaves vs True Residfaves, MSE = {pred[0]:.2f}")
plt.savefig('Scatterplot_cnn.png',
            dpi=200)
plt.show()


plt.hist(predict_df["Predicted Resid"],alpha=0.5,bins=36,label="Predicted Residfaves")
plt.hist(predict_df["True Resid"],alpha=0.5,bins=36,label="True Residfaves")
plt.legend(loc='upper left')
plt.savefig('Densities predicted_cnn.png',
            dpi=200)
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(history.history['mse'], label='MSE',color='blue')
ax1.plot(history.history['val_mse'], label = 'val_MSE',color='red')
ax2.plot(history.history['lr'], label='Learning Rate',color='green')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE')
ax2.set_ylabel('Learning Rate',color='green')
plt.title("EfficientNetB0 Top layer trained")

ax1.legend(loc='upper center')
ax2.legend(loc=0)
plt.show()

### Fine tune EfficientNet

ENB.trainable = True
model.summary()
optimizer = keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mse','mae'])
rlronp=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,
                                            patience=1, verbose=1)
history = model.fit_generator(generator=train_generator,
                                     steps_per_epoch=256,
                                     validation_data=validation_generator,
                                     validation_steps=128,
                                     epochs=10,
                                     callbacks=[rlronp,early_stopping,model_checkpoint])

test_generator.reset()
pred = model.evaluate(test_generator)


plt.plot(history.history['mse'], label='MSE')
plt.plot(history.history['val_mse'], label = 'val_MSE')

plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title("EfficientNetB0 Finetuning")
plt.legend(loc='upper right')
plt.show()

from tf_keras_vis.activation_maximization import ActivationMaximization
import matplotlib.pyplot as plt

############################################################################################################################################################
##### Source: https://github.com/christianversloot/machine-learning-articles/blob/main/visualizing-keras-model-inputs-with-activation-maximization.md  #####
##### Since documentation from the library is lack luster to say the least                                                                             #####
############################################################################################################################################################


def loss(output):
  return (output[0, 0])

def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear

# Initialize Activation Maximization
visualize_activation = ActivationMaximization(model, model_modifier)

# Generate a random seed for each activation
seed_input = tf.random.uniform((1, 224, 224, 3), 0, 255)

# Generate activations and convert into images
activations = visualize_activation(loss, seed_input=seed_input, steps=512, input_range=(30,150))
images = [activation.astype(np.float32) for activation in activations]


visualization = images[0]
plt.imshow(visualization.astype('uint8'))
plt.title("Activation Map Dense Layer")
plt.axis("off")
plt.savefig('activation.png',
            dpi=200)
plt.show()


"""   Data augmentation plots

datagen=ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2)
pic=datagen.flow(img_input,batch_size=1)
plt.figure(figsize=(16,16))
for i in range(1,10):
  plt.subplot(3, 3, i)
  batch = pic.next()
  image_ = batch[0].astype('uint8')
  plt.axis("off")

  plt.tight_layout(rect=[0, 0.03, 1, 1.5])
  plt.imshow(image_)
plt.savefig('data_augmentation.png',
            dpi=300,bbox_inches="tight")




"""