#import tensorflow as tf
#from tensorflow import keras
import numpy as np
import keras
from matplotlib import pyplot as plt
import os

'''
this is to generate a simple model to predict mnist 
it will use normal conv then use data aug on it
'''
# load data
mnist = keras.datasets.mnist
(x_train, y_train) , (x_test,y_test) = mnist.load_data()

# reshape to use conv
x_train = x_train.reshape( x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape( x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# convert class vectors to to_categorical class matrices
# example: 0 => [1,0,0,0,0,0,0,0,0] and 9 => [0,0,0,0,0,0,0,0,0,1]

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert the data from 0 to 1.0
x_train, x_test = x_train / 255.0, x_test / 255.0

model_file_name = 'saved_model.hdf5'
# if model exists, skip initial training and go to data augumentation 
if os.path.isfile(model_file_name):
  # read the model from the file
  model = keras.models.load_model(model_file_name)
else:
  # define the model and train: 
  model = keras.models.Sequential()
  model.add( keras.layers.Conv2D(32, kernel_size=(3,3), input_shape=input_shape , activation='relu' ))
  model.add( keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu' ))
  model.add( keras.layers.MaxPooling2D(pool_size=(2,2)))
  model.add( keras.layers.Dropout(rate=0.5))
  
  model.add( keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu' ))
  model.add( keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu' ))
  model.add( keras.layers.MaxPooling2D(pool_size=(2,2)))
  model.add( keras.layers.Dropout(rate=0.5))

  model.add( keras.layers.Flatten())
  model.add( keras.layers.Dense(265, activation='relu'))
  model.add( keras.layers.Dropout(rate=0.5))
  model.add( keras.layers.Dense(10, activation='softmax'))

  # compile to model
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.summary()

  #train the model with train data
  fit_history = model.fit(x_train, y_train,
    epochs=35 ,
    batch_size=200,
    validation_data=(x_test,y_test)
  )

  # save the model
  keras.models.save_model(model, model_file_name)

  # show procession of training...
  #https://github.com/sspak9/ml/blob/master/mac_plaid.ipynb
  fig = plt.figure()
  plt.plot(fit_history.history['loss'])
  plt.plot(fit_history.history['val_loss'])
  
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  fig.savefig('loss.png')
  plt.close()

  fig = plt.figure()
  plt.plot(fit_history.history['acc'])
  plt.plot(fit_history.history['val_acc'])
  
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  fig.savefig('accuracy.png')
  plt.close()

# use datagen against it
datagen = keras.preprocessing.image.ImageDataGenerator(
  rotation_range = 20,
  width_shift_range=0.15,
  height_shift_range=0.15,
  zoom_range = 0.15
)

# this will generate few parameters to be used by data gen
datagen.fit(x_train)

# check point to save
checkpt_path='models/va{val_acc:.4f}-ac{acc:.4f}-vl{val_loss:.4f}-ep{epoch:03d}.hdf5'
cp_callback=keras.callbacks.ModelCheckpoint(
  checkpt_path,
  verbose=1
)

fit_history2 = model.fit_generator(
  datagen.flow(x_train,y_train,batch_size=200),
  epochs = 150,
  validation_data = (x_test, y_test),
  callbacks=[cp_callback]
)

# show procession of training...
#https://github.com/sspak9/ml/blob/master/mac_plaid.ipynb
fig = plt.figure()
plt.plot(fit_history2.history['loss'])
plt.plot(fit_history2.history['val_loss'])
 
plt.title('model loss2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig('loss2.png')
plt.close()

fig = plt.figure()
plt.plot(fit_history2.history['acc'])
plt.plot(fit_history2.history['val_acc'])
 
plt.title('model accuracy2')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig('accuracy2.png')
plt.close()