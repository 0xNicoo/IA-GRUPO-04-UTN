from base.functions import relu, relu_prime
from base.layer import Layer
import numpy as np

class ReluLayer(Layer):
    def __init__(self, input_dim, hidden_dim):
        np.random.seed(42)
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.W = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b = np.zeros((1, hidden_dim))
        self.Z = np.array([])
        self.last_input = np.array([])

    def fordward(self, inputs):
        z = np.dot(inputs, self.W) + self.b
        self.Z = z
        self.last_input = inputs
        a = relu(z)
        return a

    def backward(self, inputs, learning_rate):
        dz = inputs * relu_prime(self.Z) 
        dW = np.dot(self.last_input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        #Acomodo los pesos y bayas
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        da = np.dot(dz, self.W.T)
        return da