import numpy as np
from base.network_builder import NetworkBuilder

x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0,1], [1,1], [0,0], [1,0]])
y_train = np.array([[0], [1], [1], [0], [1], [0], [0], [0]])

def one_hot_encode(labels, num_classes):
    labels = np.clip(labels, 0, num_classes - 1)
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels.flatten()] = 1
    return one_hot

y_train = one_hot_encode(y_train, 2)

network = NetworkBuilder.build(2, [3], 2)

network.fit(x_train, y_train, 1000, 0.1)

x_test = np.array([[0, 0]])

probabilities, predictions = network.predict(x_test)

for i in range(len(x_test)):
    print(f"Entrada: {x_test[i]}")
    print(f"Probabilidades: {probabilities[i]}")
    print(f"Predicción: {predictions[i]}")
    print()