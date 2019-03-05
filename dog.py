import numpy as np

# if using tensorflow, uncomment below. I am using plaidml as the keras backend on this machine
#from tensorflow import keras

from matplotlib import pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img

# this is the name of the dog file
dog_file_name = 'dog300.png'

# load the image
img = load_img(dog_file_name)
print('the image read in:', img)

# we need to change the image to numpy array for processing and normalize the value (0.0 to 1.0)
ar = (img_to_array(img).astype(float)) / 255.0

#show the shape 
print('array shape:', ar.shape)

# display the image in matplotlib
#%matplotlib inline
plt.figure()
plt.xticks([])
plt.yticks([])
plt.imshow(ar)
plt.show()
plt.close()

dog_datagen = ImageDataGenerator(
  rotation_range=45,
  width_shift_range=0.2,
  height_shift_range=0.2,
  zoom_range=0.3,
  horizontal_flip=True,
fill_mode='nearest')

# must convert the source a 4 tuple
ar4 = ar.reshape((1,) + ar.shape)
print('the shape of source image in 4-tuple:', ar4.shape)

# let's generate 10 modified images

nimg = 10
images = []

i = 0
for generated_image in dog_datagen.flow(ar4 ):
    # drop the 1 in front so we have image that can be shown in matplotlib
    new_image = generated_image.reshape( 300,300,3)
    images.append(new_image)
    i +=1
    if( i >= nimg):
        break

#plot the images 
fig = plt.figure( figsize = (11,2) ,dpi=96)

for i in range(nimg):
    plt.subplot(1,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i])
plt.show()
plt.close()