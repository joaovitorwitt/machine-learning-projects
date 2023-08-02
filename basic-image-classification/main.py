"""
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

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()