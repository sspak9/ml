# if using tensorflow, import keras as tf.keras
import os
import numpy as np
#import keras

import tensorflow as tf
from tensorflow import keras

import numpy as np

from tensorflow import keras
from matplotlib import pyplot as plt

# print some info
print('backend:', keras.backend.backend() ,', version:', keras.__version__, ', image_data_format:' , keras.backend.image_data_format())
is_channels_first = (keras.backend.image_data_format() == 'channels_first')

# display warning message if model already exists
if is_channels_first:
  model_file_name = 'channels_first_cnn_model.h5'
else:
  model_file_name = 'channels_last_cnn_model.h5'

if os.path.isfile(model_file_name):
  print('*** WARNING: Delete model file, if you would like to retrain ***')
  print('Model File Name:', model_file_name)
  delete_command = input('Do you wish to delete model file: ' + model_file_name + '? [Y]')
  if delete_command == 'Y' or delete_command =='':
    os.remove(model_file_name)
    is_mode_exists = False
    print('model file has been deleted:', model_file_name)
        
if os.path.isfile(model_file_name):
  is_model_exists = True
else:
  is_model_exists = False

	
# get mnist data
mnist = keras.datasets.mnist

print('loading MNIST data...')
# using path saves to the ~/.keras/data/path location so it's not downloaded next time
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#show the "shape" of downloaded data
print('train data size:', x_train.shape)
print('train label (expected) value size:', y_train.shape)
print('test data size:', x_test.shape)
print('test expected value:',y_test.shape)

print('\n\ndisplaying few training samples')

#function to copy 1 mage to larger image map
def copy_image(target , ty, tx, src):
  for y in range(28):
    for x in range(28):
      target[ty*28+y][tx*28+x] = src[y][x]
  return target

# show 20 x 20
ysize = 20
xsize = 20
start_offset = 400
base_index = start_offset +(ysize * xsize)

print('MNIST data. Offset:', start_offset)

image = np.zeros((28*ysize, 28*xsize), dtype=np.int)

for y in range(ysize):
  for x in range(xsize):
    index = y*xsize + x
    src = x_train[index + base_index]
    image = copy_image(image , y ,x , src)

#%matplotlib inline
from matplotlib import pyplot as plt
plt.figure(figsize=(7,7))
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.imshow(image , cmap='gray_r')
plt.show()
plt.close()

# shape data for CNN

# convert the shape of data depending on the image data format
is_channels_first = (keras.backend.image_data_format() == 'channels_first')

if is_channels_first :
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

print('is model exists:', is_model_exists)
if not is_model_exists:
  # setup model and compile
  model = keras.models.Sequential()
  model.add( keras.layers.Conv2D(32, kernel_size=(3,3), input_shape=input_shape , activation='relu' ))
  model.add( keras.layers.MaxPooling2D(pool_size=(2,2)))
  model.add( keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu' ))
  model.add( keras.layers.MaxPooling2D(pool_size=(2,2)))
  model.add( keras.layers.Dropout(0.5))
  model.add( keras.layers.Flatten())
  model.add( keras.layers.Dense(265, activation='relu'))
  model.add( keras.layers.Dropout(0.5))
  model.add( keras.layers.Dense(10, activation='softmax'))

  # compile to model
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  # show summary
  model.summary()
  #train the model with train data
  fit_history = model.fit( x_train2, 
                      y_train2, 
                      epochs=35 , 
                      batch_size=200, 
                      validation_data=(x_test2,y_test2))
  print('*** Writing to model file:', model_file_name)
  model.save(model_file_name)
else:
  print('*** Did not train as model exists ***')

if not is_model_exists:
  # show procession of training...
  plt.plot(fit_history.history['loss'])
  plt.plot(fit_history.history['val_loss'])

  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

  plt.plot(fit_history.history['acc'])
  plt.plot(fit_history.history['val_acc'])

  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

# predict for my test data
print('*** Reading from saved model:', model_file_name)
model = keras.models.load_model(model_file_name)

predictions = model.predict(x_test2)

my_matrix = np.zeros( (10,10), dtype='int')

# count of good guesses
count_matrix = np.zeros( (10,), dtype='int')
good_matrix = np.zeros( (10,), dtype='int')

# iterate through 10,000 test data
for i in range(10000):
  count_matrix[y_test[i]] +=1
  guess = np.argmax(predictions[i])
  if guess == y_test[i]:
    good_matrix[guess] +=1
  else:
    # increment [expected][guess] matrix
    my_matrix[y_test[i]][guess] += 1

# show good matrix
print('Good guesses:')
for i in range(10):
  percent = "( {:.2f}".format((good_matrix[i] * 100.0) / count_matrix[i]) + " %)"
  print('match count for:',i,'=', good_matrix[i] , '/',count_matrix[i] , percent)

print('\nConfusion Matrix')

fig = plt.figure()
plt.xticks( range(10))
plt.yticks( range(10))

for y in range(10):
  for x in range(10):
    if my_matrix[y][x] != 0:
      # put text
      plt.text( x-len(str(x)) * 0.2, y+0.1, str(my_matrix[y][x]))

plt.xlabel('prediction')
plt.ylabel('expected')
plt.imshow(my_matrix, cmap='YlOrRd')
plt.colorbar()
plt.show()
plt.close()

non_match_list = []
for i in range(10000):
  if y_test[i] == 9:
    guess = np.argmax(predictions[i])
    if guess == 4:
        non_match_list.append(i)

fig = plt.figure( figsize = (10,2))

for i in range(len(non_match_list)):
  plt.subplot(1,20,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  index = non_match_list[i]
  plt.imshow(x_test[index], cmap='gray_r')
plt.show()
plt.close()
