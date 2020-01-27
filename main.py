# COSC 525 Deep Learning - Project 1
# Aaron Wilson & Bohan Li
# Due Jan 28 2020

import numpy as np, sys
from neuralnet import NeuralNetwork
import matplotlib.pyplot as plt


def main():

    # script = str(sys.argv[1])
    script = 'xor'

    if script == "example":

        x = np.array([[0.05, 0.1]])
        y = np.array([[0.01, 0.99]])

        parameters = {
            "num inputs" : 2,
            "num outputs" : 2,
            "num hidden layers" : 1,
            "num neurons" : 2,
            "activations": ['sigmoid'] * 2,
            "learning rate": 0.5,
            "loss function": 'squared error'
        }

        nn = NeuralNetwork(parameters)

        # In this example, don't use randomly-initialized weights - use the ones from the example. So, to make things
        # simple, just over-write the existing, randomly-initialized weights from the NeuralNetwork instantiation.
        weights = np.array([[0.15, 0.20, 0.25, 0.30], [0.40, 0.45, 0.50, 0.55]])
        biases = np.array([[0.35, 0.35], [0.60, 0.60]])

        for l in range(0, len(nn.layers)):
            nn.layers[l].weights = weights[l]
            nn.layers[l].biases = biases[l]

        nn.layers[0].neurons[0].weights = weights[0, 0:2]
        nn.layers[0].neurons[1].weights = weights[0, 2::]
        nn.layers[0].neurons[0].bias = biases[0, 0]
        nn.layers[0].neurons[1].bias = biases[0, 1]

        nn.layers[1].neurons[0].weights = weights[1, 0:2]
        nn.layers[1].neurons[1].weights = weights[1, 2::]
        nn.layers[1].neurons[0].bias = biases[1, 0]
        nn.layers[1].neurons[1].bias = biases[1, 1]

        out = nn.train(x, y)

    elif script == "and":

        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])

        parameters = {
            "num inputs": 2,
            "num outputs": 1,
            "num hidden layers": 0,
            "num neurons": 0,
            "activations": ['sigmoid'],
            "learning rate": 0.5,
            "loss function": 'squared error'
        }

        nn = NeuralNetwork(parameters)

        out = nn.train(x, y)

    elif script == "xor":

        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])

        parameters_perceptron = {

            "num inputs": 2,
            "num outputs": 1,
            "num hidden layers": 0,
            "num neurons": 0,
            "activations": ['sigmoid'],
            "learning rate": 0.5,
            "loss function": 'squared error'
        }

        parameters_multilayer_perceptron = {

            "num inputs": 2,
            "num outputs": 1,
            "num hidden layers": 1,
            "num neurons": 2,
            "activations": ['sigmoid'] * 2,
            "learning rate": 0.1,
            "loss function": 'squared error'

        }

        nn_single = NeuralNetwork(parameters_perceptron)
        nn_single.train(x, y, epochs=10)

        nn_multi = NeuralNetwork(parameters_multilayer_perceptron)
        nn_multi.train(x, y, epochs=10)


    plt.plot(nn_multi.loss_epoch)
    plt.show()



if __name__ == "__main__":

    main()