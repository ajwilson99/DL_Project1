# COSC 525 Deep Learning - Project 1
# Aaron Wilson & Bohan Li
# Due Jan 28 2020

import numpy as np, sys
from neuralnet import Neuron, FullyConnectedLayer


def main():

    x = np.array([1, 0])
    tmp = FullyConnectedLayer(num_neurons=3, activation="sigmoid", num_inputs=2, learning_rate=0.1)
    out = tmp.calculate(x)


if __name__ == "__main__":

    main()