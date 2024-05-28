from base.network import Network
from base.softmax_layer import SoftmaxLayer
from base.relu_layer import ReluLayer

class NetworkBuilder():
    def build(input_layer: int, hidden_layers: list[int], output_layers: int):
        network = Network()
        network.add_layer(ReluLayer(input_layer, hidden_layers[0]))
        for i in range(len(hidden_layers[1:])):
            network.add_layer(ReluLayer(hidden_layers[i], hidden_layers[i-1]))
        network.add_layer(SoftmaxLayer(hidden_layers[-1], output_layers))
        return network