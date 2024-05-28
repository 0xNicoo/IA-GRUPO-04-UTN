class Layer():
    def fordward(self, inputs):
        return NotImplementedError

    def backward(self, inputs, learning_rate):
        return NotImplementedError