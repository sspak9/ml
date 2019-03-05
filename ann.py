# if using tensorflow, import keras as tf.keras
import numpy as np
#import keras

import tensorflow as tf
from tensorflow import keras

print('keras version:', keras.__version__)

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
start_offset = 0
base_index = start_offset * (ysize * xsize)

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

# reshape to use flat array instead of 28x28

x_train_reshaped = x_train.reshape(x_train.shape[0],784)
x_test_reshaped = x_test.reshape(x_test.shape[0], 784)

x_train_reshaped = x_train_reshaped.astype('float32')
x_test_reshaped = x_test_reshaped.astype('float32')

x_train_reshaped /= 255.0
x_test_reshaped /= 255.0

y_hot_train = keras.utils.to_categorical(y_train, num_classes=10)
y_hot_test = keras.utils.to_categorical(y_test, num_classes=10)

model = keras.models.Sequential()
# input layer is just 784 inputs coming in as defined in the hidden layer below

# hidden layer
model.add( keras.layers.Dense(512, input_shape=(784,), activation='relu'))

#output layer
model.add( keras.layers.Dense(10, activation='softmax'))

# compile to model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#train the model with train data
fit_history = model.fit(x_train_reshaped, y_hot_train,
  epochs=25 ,
  batch_size=200,
  validation_data=(x_test_reshaped,y_hot_test)
)

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
plt.close()

# model 2 with drop output

model2 = keras.models.Sequential()
# input layer is just 784 inputs coming in as defined in the hidden layer below

# hidden layer
model2.add( keras.layers.Dense(512, input_shape=(784,), activation='relu'))
model2.add( keras.layers.Dropout(rate=0.5))

#output layer
model2.add( keras.layers.Dense(10, activation='softmax'))

# compile to model
model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model2.summary()

#train the model with train data
fit_history2 = model2.fit(x_train_reshaped, y_hot_train,
  epochs=35 ,
  batch_size=200,
  validation_data=(x_test_reshaped,y_hot_test)
)

# show procession of training...
plt.plot(fit_history2.history['loss'])
plt.plot(fit_history2.history['val_loss'])
 
plt.title('model2 loss ')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(fit_history2.history['acc'])
plt.plot(fit_history2.history['val_acc'])
 
plt.title('model2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.close()

# confusion matrix
my_matrix = np.zeros( (10,10), dtype='int')

# predict for my test data
predictions = model2.predict(x_test_reshaped)

# iterate through 10,000 test data
for i in range(10000):
    guess = np.argmax(predictions[i])
    if guess != y_test[i]:
        # increment [expected][guess] matrix
        my_matrix[y_test[i]][guess] += 1

# show
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

# show 2's that are being guessed as 7'samples
non_match_list = []
for i in range(10000):
    if y_test[i] == 2:
        guess = np.argmax(predictions[i])
        if guess == 7:
            non_match_list.append(i)

fig = plt.figure( figsize = (5,10))

for i in range(len(non_match_list)):
    plt.subplot(1,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    index = non_match_list[i]
    plt.imshow(x_test[index], cmap='gray_r')
plt.show()
plt.close()

