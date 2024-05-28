import numpy as np
from base.layer import Layer

class Network():
    def __init__(self):
        self.layers= []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def fit(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):

            layer_output = self.layers[0].fordward(x_train)
            for layer in self.layers[1:]:
                layer_output = layer.fordward(layer_output)

            loss = self.cross_entropy_loss(y_train, layer_output)

            reverse_layers = self.layers[::-1]
            layer_output_backprop = reverse_layers[0].backward(y_train, learning_rate)
            for layer in reverse_layers[1:]:
                layer_output_backprop = layer.backward(layer_output_backprop, learning_rate)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        layer_output = self.layers[0].fordward(X)
        for layer in self.layers[1:]:
            layer_output = layer.fordward(layer_output)
        return layer_output, np.argmax(layer_output, axis=1)

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m