import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # return numpy arrays

# print(train_images.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     # plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# setting up neuro layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=50)

# evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('loss ', test_loss, ' accuracy: ', test_acc)

# prediction = model.predict(test_images[0:1])

prediction = model.predict(np.expand_dims(test_images[0], 0))

print(class_names[np.argmax(prediction)])


