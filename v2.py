import os
import numpy as np
#import keras

import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist

print('loading MNIST data...')

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# convert the shape of data depending on the image dataformat
is_channels_first = (keras.backend.image_data_format() == 'channels_first')

if is_channels_first:
    x_train2 = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test2 = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    x_train2 = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test2 = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

# get hot label output

y_train2 = keras.utils.to_categorical(y_train, num_classes=10)
y_test2 = keras.utils.to_categorical(y_test, num_classes=10)

# normalize the data
x_train2 = x_train2.astype('float32')
x_test2 = x_test2.astype('float32')

# convert the data from 0 to 1.0
x_train2, x_test2 = x_train2 / 255, x_test2 / 255

print('x train shape:',x_train2.shape)
print('y train shape:',y_train2.shape)
print('x test shape:',x_test2.shape)
print('y test shape:',y_test2.shape)



model = keras.models.Sequential()

model.add( keras.layers.Conv2D(32, kernel_size=(3,3), input_shape=input_shape , activation='relu' ))
model.add( keras.layers.Dropout(0.1))
model.add( keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu' ))
model.add( keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add( keras.layers.Dropout(0.2))

model.add( keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu' ))
model.add( keras.layers.Conv2D(64, kernel_size=(3,3),   activation='relu' ))
model.add( keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add( keras.layers.Dropout(0.5))

model.add( keras.layers.Flatten())
model.add( keras.layers.Dense(256,activation='relu'))
model.add( keras.layers.Dropout(0.5))
model.add( keras.layers.Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()

# start with some basics set
model.fit( x_train2, 
			y_train2, 
			epochs=2 , 
			batch_size=128, 
			validation_data=(x_test2,y_test2))
 
datagen = keras.preprocessing.image.ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.10,
  height_shift_range=0.10,
  zoom_range=0.15)

# allow datagen to get some metrics over the train data
datagen.fit(x_train2)

checkpt_path='models3/va{val_acc:.4f}-e{epoch:04d}-a{acc:.5f}-l{loss:.5f}-vl{val_loss:.5f}.hdf5'

cp_callback=keras.callbacks.ModelCheckpoint(
  checkpt_path,
  verbose=1,
)
fit_history = model.fit_generator( datagen.flow( x_train2, y_train2, batch_size=128),
    steps_per_epoch=len(x_train) / 128,
    epochs = 600,
    validation_data = (x_test2, y_test2)
    ,callbacks=[cp_callback]
)

# save val_acc to disk
np.save('loss',fit_history.history['loss'])
np.save('val_loss',fit_history.history['val_loss'])
np.save('acc',fit_history.history['acc'])
np.save('val_acc',fit_history.history['val_acc'])

# show procession of training...

from matplotlib import pyplot as plt
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('loss.png')
plt.close()

plt.plot(fit_history.history['acc'])
plt.plot(fit_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('accuracy.png')
plt.close()


	