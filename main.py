# COSC 525 Deep Learning - Project 1
# Aaron Wilson & Bohan Li
# Due Jan 28 2020

import numpy as np, sys
from neuralnet import NeuralNetwork


def main():

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    num_neurons = 3
    num_hidden_layers = 3
    hidden_activations = 'linear'
    output_activation = 'sigmoid'
    activations = [hidden_activations]*num_hidden_layers + [output_activation]
    num_inputs = x.shape[1]
    learning_rate = 0.1
    loss_function = 'squared error'

    tmp = NeuralNetwork(num_hidden_layers, num_neurons, activations, 2, loss_function, 0.1)

    out = tmp.train(x[-1], y[-1])

    a = 1


if __name__ == "__main__":

    main()