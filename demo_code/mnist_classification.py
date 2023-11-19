
"""
Written & Designed By Afonso Moniz Moreira V1.00 21052020
"""


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




mnist=tf.keras.datasets.mnist #28x28 images of hand-written digits 0-9
(x_train,y_train),(x_test,y_test) = mnist.load_data() #Dividing the set into training and testing

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)



print(x_train[0]) #This is a Tensor
plt.imshow(x_train[0], cmap=plt.cm.binary)



# Data normalization improves significantly the DNN  performance#
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=np.array(tf.keras.utils.normalize(x_test,axis=1))
x_test.shape


model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 1st hidden layer with 128 neurons
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 2nd hidden layer with 128 neurons
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Output hidden layer with 10 neurons

model.compile(optimizer='SGD',loss='binary_crossentropy', metrics=['accuracy'])
history=model.fit(x_train,y_train, epochs=3, validation_data=(x_test,y_test))


val_loss, val_acc=model.evaluate(x_test, y_test)

model.summary()


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()









