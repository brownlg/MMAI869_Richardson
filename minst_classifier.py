
# first attempt at an image classifier
from sklearn import datasets
import tensorflow as tf

#dimensions of input image
WINDOW_X = 28
WINDOW_Y = 28

from keras import backend as K
#print(K.tensorflow_backend._get_available_gpus())
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], WINDOW_X, WINDOW_Y, 1)
x_test = x_test.reshape(x_test.shape[0], WINDOW_X, WINDOW_Y, 1)
input_shape = (WINDOW_Y, WINDOW_X, 1)

#create y_train and y_test values
x_test = x_test / 255
x_train = x_train / 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(WINDOW_X * WINDOW_Y, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10 ,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=10)

print(model.evaluate(x_test, y_test))
