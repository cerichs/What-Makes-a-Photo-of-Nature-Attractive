# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:25:35 2022

@author: Corne
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.preprocessing.image import img_to_array


np.random.seed(20)

# Load the data

train_df=pd.read_pickle("train_df.pkl")
test_df=pd.read_pickle("test_df.pkl")
#train_df=pd.read_pickle("df_train_standard.pkl")
#test_df=pd.read_pickle("df_test_standard.pkl")

# Data augmentation

datagen = ImageDataGenerator(rescale=1./255,validation_split=0.1,
                             rotation_range=90,
                             width_shift_range=0.2,
                             height_shift_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator=datagen.flow_from_dataframe(
    dataframe=train_df, directory="../images", 
    x_col="path", y_col="standard", seed = 20,
    class_mode="raw", color_mode='rgb', target_size=(224,224), batch_size=64, subset = "training")

validation_generator = datagen.flow_from_dataframe(dataframe=train_df, directory="../images", 
    x_col="path", y_col="standard", seed = 20,
    class_mode="raw",color_mode='rgb', target_size=(224,224), batch_size=64, subset = "validation")


test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, directory="../images",
    x_col="path", y_col="standard",seed = 20, class_mode="raw", target_size=(224,224), 
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



# Define model

keras.backend.clear_session()

model = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(224, 224, 3)),

    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(128, kernel_size=(3, 3),
                        activation='relu'),
    
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(128, kernel_size=(3, 3),
                        activation='relu'),
    
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(256, kernel_size=(3, 3),
                        activation='relu'),
    keras.layers.Conv2D(256, kernel_size=(3, 3),
                        activation='relu'),
    
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(256, kernel_size=(3, 3),
                        activation='relu'),
    
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    #keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(1,activation="linear")   
])
rlronp=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,
                                            patience=1, verbose=1)


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
                                     callbacks=[rlronp,early_stopping,model_checkpoint])
#model.save_weights("model.h5")

# Evaluate model on test data
test_generator.reset()
pred = model.evaluate(test_generator)

def get_predicts(model,test_df):
    temp,temp_path,temp_true=[],[],[]
    temp_df=test_df.reset_index(drop=True)
    for i in range(len(temp_df)):
        try:
            img = Image.open(temp_df.iloc[i,2])
            img_input = np.expand_dims(img_to_array(img)/255, axis=0)
            temp.append(model.predict(img_input)[0][0])
            temp_path.append(temp_df.iloc[i,2])
            temp_true.append(temp_df.iloc[i,6])
            if i%1000==0:
                print(i)
        except:
            pass
    predict_df=pd.DataFrame()
    predict_df["Image path"]=temp_path
    predict_df["Predicted Resid"]=temp
    predict_df["True Resid"]=temp_true
    predict_df.to_pickle("predicted_df_cnnn_standard.pkl")
    return predict_df

predict_df=get_predicts(model,test_df)

fig, ax1 = plt.subplots()
ax1.plot(history.history['mse'], label='MSE',color='blue')
ax1.plot(history.history['val_mse'], label = 'val_MSE',color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE')
ax1.set_xlim(1,38)
ax1.set_ylim(0.6,1)
plt.title("CNN MSE = 0.694")

ax1.legend(loc='upper center')
plt.savefig('CNN_training_from_1.png',
            dpi=200)
plt.show()



from tf_keras_vis.activation_maximization import ActivationMaximization
import matplotlib.pyplot as plt

############################################################################################################################################################
##### Source: https://github.com/christianversloot/machine-learning-articles/blob/main/visualizing-keras-model-inputs-with-activation-maximization.md  #####
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
plt.imshow(visualization.astype('uint8'),cmap="gray")
plt.show()




def visualize_conv_layer(layer_name,img):
  
  ###############################################################################
  ##### Source: https://valueml.com/get-the-output-of-each-layer-in-keras/  #####
  ###############################################################################
  
  layer_output=model.get_layer(layer_name).output  #get the Output of the Layer

  intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output) #Intermediate model between Input Layer and Output Layer which we are concerned about

  intermediate_prediction=intermediate_model.predict(img) #predicting in the Intermediate Node
  
  row_size=8
  col_size=8
  
  img_index=0

  print(np.shape(intermediate_prediction))
    #---------------We will subplot the Output of the layer which will be the layer_name----------------------------------#
  
  fig,ax=plt.subplots(row_size,col_size,figsize=(20,16)) 
  
  
  for row in range(0,row_size):
    for col in range(0,col_size):
      ax[row][col].imshow(intermediate_prediction[0, :, :, img_index])
      ax[row][col].axis("off")
      img_index=img_index+1 #Increment the Index number of img_index variable
  fig.tight_layout()
  plt.savefig('1stconv_robin.png',
              dpi=200)
      
img=Image.open("C:/Users/Corne/Documents/GitHub/02466Fagprojekt/images/49693844557_224103a114_z.jpg")
x = np.asarray(img, np.int)
#x=np.expand_dims(img,axis=0)
x = x.reshape((1,) + x.shape) 
#x =x/ 255.
visualize_conv_layer('conv2d',x)

