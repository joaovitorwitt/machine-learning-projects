"""
    Tutorial url: https://www.tensorflow.org/tutorials/keras/classification?hl=en
    train a model to classify images of clothing, like sneakers and shirts
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)


"""
importing the MNIST dataset

contains 70000 grayscale images in 10 categories. (28x28 pixels)
"""

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/pop', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


"""
Preprocess the data

The data must be preprocessed before training the network.
Currently the pixels range from 0 to 255
"""

plt.figure()
plt.imshow(train_images[3])
plt.colorbar()
plt.grid(False)
plt.show()

"""
Scale these values to a range of 0 to 1 before feeding them to the neural network model.
We simply divide by 255.0
"""

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure()
plt.imshow(train_images[3])
plt.colorbar()
plt.grid(False)
plt.show()

"""
verify that the data is in the correct form.
Lets display the first 25 images from the training set and display the class name below each image
"""

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


"""
Build the model

building the neural network requires configuring the layers of the model
then compiling the model.
"""


"""
Set up the layers

layers are the basic building block of a netural network.
layers extract meaningful representations from the data feed into them.
"""

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

"""
the first layer transforms the format of the images from two dimensional
to one dimensional (28 * 28 = 784px)

the first Dense layer has 128 nodes (or neurons).

the second Dense layer returns a logits array with length of 10
Each node contains a score that indicates the current image class
"""


"""
Compile the model

Before we compile the model we need:

Loss function: measures how accurate the model is duruing training
Optimizer: How the model is updated based on the data it sees and its loss function.
Metrics: used to monitor the training and testing steps.
"""

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


"""
Train the model

to being training the model

1st - feed the training data to the model.
2nd - the model learns to associate images and labels
3rd - ask the model to make predictions about a test set
4th - verify if the predictions match the labels from the test_labels array
"""

"""
Feed the model
"""
model.fit(train_images, train_labels, epochs=10)


"""
Evaluate accuracy
"""
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest Accuracy: {test_acc}")

"""
accuracy on the training dataset: 0.9106
accuracy on the testing dataset: 0.8860

The accuracy on the test dataset is a little less than the accuracy on the training dataset
This gap represents overfitting.
Overfitting happens when the model performs worse on new data. (ig: testing data)
"""


"""
Make predictions

First attach a softmax layer to convert the models linear outputs to probabilities
"""

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions =   probability_model.predict(test_images)

print(f"MODEL PREDICTION: {predictions[0]}")

"""
A prediction is an array of 10 numbers.
They represent the models confidence that the image corresponds to each of the 10 different articles of clothing.
"""

"""
we can see which labels has the highest confidence value
"""

print(f"HIGHEST CONFIDENCE VALUE: {np.argmax(predictions[0])}")

"""
the model is most confident that this image is an ankle boot
"""

print(f"ACTUAL LABEL: {test_labels[0]}")


"""
Graph this to look at the full set of 10 class predictions
"""

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i] 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


"""
Verifying predictions looking at the 0th image, predictions, predictions_array

Correct predictions labels are blue and incorrect predictions labels are red
"""

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

"""
Now lets look at the 12th image
"""

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

"""
plot several images
the model can be wrong even when very confident
"""

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


"""
Use the trained model
"""

img = test_images[0]
print(img.shape)

"""
tf.keras models are optimized to make predictions on collections of data.
even though it is a single image, we need to add to a list.
"""

img = (np.expand_dims(img, 0))
print(img.shape)

"""
now predict the correct label for this image
"""

predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()