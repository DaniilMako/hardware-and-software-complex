import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigm_deriv(x):
    return x*(1 - x)

def loss(pred, y):
    return (pred - y)

class Neuron:
    def __init__(self):
        self._weights = np.random.randn(2)
        self.bias = np.random.randn()

    def forward(self, x):
        res = np.dot(x, self._weights) + self.bias
        return sigmoid(res)

    def backward(self, x):
        return sigm_deriv(x)

class Model:
    def __init__(self):
        self.neurons = [Neuron(), Neuron(), Neuron()]
        self.learn = 0.15

    def forward(self, x):
        hidden_res = np.array([self.neurons[0].forward(np.array(x)), self.neurons[1].forward(np.array(x))])
        output = self.neurons[2].forward(hidden_res)
        return output

    def backward(self, x, err):
        d_out = err * self.neurons[2].backward(self.forward(x))
        err0 = d_out * self.neurons[2]._weights[0]
        err1 = d_out * self.neurons[2]._weights[1]
        d_hid0 = err0 * self.neurons[0].backward(self.neurons[0].forward(x))
        d_hid1 = err1 * self.neurons[1].backward(self.neurons[1].forward(x))

        self.neurons[2]._weights += np.array([self.neurons[0].forward(np.array(x)), self.neurons[1].forward(np.array(x))]) * d_out * self.learn
        self.neurons[2].bias += np.sum(d_out) * self.learn 
        self.neurons[0]._weights += np.array(x) * d_hid0 * self.learn
        self.neurons[0].bias += np.sum(d_hid0) * self.learn
        self.neurons[1]._weights += np.array(x) * d_hid1 * self.learn
        self.neurons[1].bias += np.sum(d_hid1) * self.learn

model = Model()

X = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for epoch in range(15000):
    for x, y in X:
        output = model.forward(x)
        err = loss(y, output)
        model.backward(x, err)

for x, _ in X:
    output = model.forward(x)
    print("Input: {0}, Res: {1}".format(x, round(output)))