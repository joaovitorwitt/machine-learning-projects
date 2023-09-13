import tensorflow as tf
import numbers as np
from tensorflow import Sequential
from keras import Sequential
from keras.layers import Dense

"""
Dense layer means a set of fully connected neurons, every neuron is connected to every other neuron in the next layer.
"""
l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean-squared-error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

model.fit(xs, ys, epochs=500)


"""
The term 'prediction' is used when we are dealing with a certain amount of uncertainty
"""

print(model.predict([10.0]))
print("Here is what I learned: {}".format(l0.get_weights()))