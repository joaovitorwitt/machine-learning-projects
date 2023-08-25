

"""
    Tutorial url: https://www.tensorflow.org/tutorials/keras/classification?hl=en
    train a model to classify images of clothing, like sneakers and shirts
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)


"""
    import the Fashion MNIST dataset
    this dataset contains 70000 grayscale images in 10 categories (28x28 pixels)

    60000 images are used to train the model and 10000 are used to evaluate the model
"""

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


"""
    the names of the classes are not included in the dataset, so lets include them
"""

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(train_images.shape)
# print(len(train_images))
# print(len(test_images))


"""
    Preprocess the data
    first the data must be preprocessed before training the network
    we can see the pixels value ranging from 0 to 255
"""

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid()
plt.show()

"""
    lets scale those values to range from 0 to 1 dividing by 255
    the training and testing set should be preprocessed in the same way
"""

train_images = train_images / 255.0
test_images = test_images / 255.0


"""
    lets verify if the data is in correct format displaying the first 25 images with the class name below
"""

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


"""
    Build the model
    first we need to configure the layers
"""

"""
    layers extract representations from the data
    these representations must be meaningful for the problem
"""
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10)])

"""
    the first layer transforms the format of the images from a two-dimensional array (28x28)
    to a one-dimensional array (28*28 = 784)
"""

"""
    Compile the model

    before compiling the model we need a loss function, a optimizer, metrics

    Loss Function = measures how accurate the model is during training
    Optimizer = how the model is updated based on the data it sees and the loss function
    Metrics = type of metrics used to monitor the training and testing steps
"""

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])


"""
    Train the model

    1st - feed the training data into the model
    2nd - the models learns to associate images and labels
    3rd - ask the model to make predictions about a test set
    4th - verify that the predictons match the labels from a test label
"""

model.fit(train_images, train_labels, epochs=10)

"""
    the model reaches about 90% accuracy on the training data
"""

"""
    Next we need to evaluate the accuracy
"""

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f"\nTest accuracy: {test_acc}")