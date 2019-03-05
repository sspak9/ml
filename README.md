# Tensorflow.js based Javascript UI to predict hand-written digits
This is a simple JavaScript UI based app that downloads the Python based Keras trained model to predict the hand-drawn digit. This was converted from a "quick and dirty" learning project that leveraged flask python web server that accepted drawing from a browser and returned a guess with percentages.

To run from your local dev machine, get the code, and do below to launch a web server at localhost:3000
```bash
npm install
npm start
```

Use any HTML5 capable browser ( anything other than IE) to go to http://localhost:3000 to view the UI below.

I've deployed this to a Heroku URL:
https://sspak9-ml.herokuapp.com/


Draw with a white brush and erase with a black brush. When done, click Guess to predict the digit and show the percentage assigned for various digits the Kera ML trained model predicted.

![screen](images/screen.png)

### How the model was trained
The model was trained offline using python based tensorflow/keras framework against MNIST data.  The model selected leverages convolutional network that gave about 99.61% against the "test" data that were not used for training

The actual python code used to train the model is shown below. I captured the snapshot of the model at the end of each epoch. I picked the one with the highest validation accuracy and converted that into tensorflow.js format using the command:

```bash
tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```
The two files generated were copied to /public/model folder

### How index.html "enables" the tensorflow.js

I download the tensorflow.js library first in the index.html file
```html
<head>
<meta charset="utf-8"/>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/tensorflow/0.15.2/tf.min.js"></script>
</head>
```
Then the JavaScript on the same page downloads the model:
```javascript
// load model
var model;

mp = tf.loadLayersModel('/model/model.json');
mp.then( function(value) {
  model = value;
})
```
BTW, I haven't figured out the Promise yet correctly, so please execuse any blunder in using it.

When the Guess button is clicked, the drawn digit on the left canvas is down-sampled to 28x28 gray image and is shown on the right.

Then the gray image that is of 28x28 size (784) is converted to float array of value between 0 to 1.0 and then submitted to the model to predict

```javascript
// convert to 4D tensor
  let t4d = tf.tensor4d( fa , [1 , 28, 28, 1]);
  
  // predict
  let rp = model.predict(t4d)
```
The rest of the code is used to extract the data from the Promise and display on the UI.



[View Jupyter Notebook on simple ANN model](Keras_ANN.ipynb)

[View Jupyter Notebook on sample cnn model](mac_plaid.ipynb)

[View Jupyter Notebook on using data augmentation to enhance the cnn model](CNN_augmentation.ipynb)

Sample python code to train on MNIST data using tensorflow/kera
```python
import time
import os

#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#import keras

import tensorflow as tf
from tensorflow import keras
import numpy as np


# helper function to create unique sub folder
def create_folder(folder_name):
  if (not os.path.exists(folder_name)):
    os.makedirs(folder_name)
  new_dir = folder_name + "/{}".format(time.time())
  if (not os.path.exists(new_dir)):
    os.makedirs(new_dir)
  return new_dir
 
# get mnist data
mnist = keras.datasets.mnist

# using path saves to the ~/.keras/data/path location so it's not downloaded next time
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#show the "shape" of downloaded data
print('train data size:', x_train.shape, 'train expected value:', y_train.shape, 'test data size:', x_test.shape , 'test expected value:',y_test.shape)

# reshape to fit conv map
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# convert the data from 0 to 1.0
x_train, x_test = x_train / 255.0, x_test / 255.0

# define the model: 
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
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#setup dashboard callback
#logs_dir = create_folder('logs')
#tf_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

# model snapshot directory
models_dir = create_folder('models2')

checkpt_path=models_dir+'/va{val_acc:.4f}-ac{acc:.4f}-vl{val_loss:.4f}-ep-{epoch:03d}-.hdf5'
cp_callback = keras.callbacks.ModelCheckpoint(
  checkpt_path ,
  verbose=1
)

#train the model with train data
fit_history = model.fit(x_train, y_train,
  epochs=100 ,
  batch_size=200,
  validation_data=(x_test,y_test)
  ,callbacks=[ cp_callback]
)

```