# COSC 525 Deep Learning - Project 1
# Aaron Wilson & Bohan Li
# Due Jan 28 2020

import numpy as np, sys
from neuralnet import NeuralNetwork
import matplotlib.pyplot as plt


def main():

    script = str(sys.argv[1])

    # script = 'example'

    if script == "example":

        x = np.array([[0.05, 0.1]])
        y = np.array([[0.01, 0.99]])

        desired_y = np.array([[0.773, 0.778]])  # After one training step (epoch)

        parameters = {
            "num inputs" : 2,
            "num outputs" : 2,
            "num hidden layers" : 1,
            "num neurons" : 2,
            "activations": ['sigmoid'] * 2,
            "learning rate": 0.5,
            "loss function": 'binary cross entropy'
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

        nn.train(x, y, epochs=1)

        new_output = nn.calculate(x)

        # Print to screen
        print('\n')
        print('Class example: single epoch')
        print('--------------------------------------------------------------------------------')
        print('Initial output is: {}. After one training step, the next output should be {}, (with sigmoid activation).'.format(y, desired_y))
        print('New output is... {}'.format(new_output))

    elif script == "and":

        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])

        parameters = {
            "num inputs": 2,
            "num outputs": 1,
            "num hidden layers": 0,
            "num neurons": 0,
            "activations": ['sigmoid'],
            "learning rate": 0.1,
            "loss function": 'squared error'
        }

        nn = NeuralNetwork(parameters)

        nn.train(x, y, epochs=2000)

        # Plotting
        plt.plot(nn.loss_epoch)
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('AND Neural Network, {} activation and \n {} loss'.format(parameters['activations'][0], parameters['loss function']))
        plt.show()

        # Printing to screen
        print('\n')
        print('AND Gate Perceptron:')
        print('--------------------------------------------------------------------------------')
        print('For y = 1, output should be >= 0.5. For y = 0, output should be < 0.5.')
        for input, output in zip(x, y):
            print("MULTI-LAYER PERCEPTRON: Output for input x = {} should be y = {}. Computed value: {}".format(input,
                                                                                                                output,
                                                                                                                nn.calculate(
                                                                                                                    input)))

    elif script == "xor":

        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])

        parameters_perceptron = {

            "num inputs": 2,
            "num outputs": 1,
            "num hidden layers": 0,
            "num neurons": 0,
            "activations": ['sigmoid'],
            "learning rate": 0.1,
            "loss function": 'squared error'
        }

        parameters_multilayer_perceptron = {

            "num inputs": 2,
            "num outputs": 1,
            "num hidden layers": 1,
            "num neurons": 4,
            "activations": ['sigmoid'] * 2,
            "learning rate": 0.1,
            "loss function": 'squared error'

        }

        nn_single = NeuralNetwork(parameters_perceptron)
        nn_single.train(x, y, epochs=2000)

        nn_multi = NeuralNetwork(parameters_multilayer_perceptron)
        nn_multi.train(x, y, epochs=2000)

        # Plotting
        plt.plot(nn_single.loss_epoch, label='Single Perceptron')
        plt.ylabel('Loss')
        plt.plot(nn_multi.loss_epoch, label='Multi-layer Perceptron')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.title('XOR Neural Network {} activation and \n {} loss'.format(
            parameters_multilayer_perceptron['activations'][0], parameters_multilayer_perceptron['loss function']))
        plt.show()

        # Printing results to screen
        print('\n')
        print('XOR Gate: Single-layer Perceptron:')
        print('--------------------------------------------------------------------------------')
        print('For y = 1, output should be >= 0.5. For y = 0, output should be < 0.5.')
        for input, output in zip(x, y):
            print("SINGLE PERCEPTRON: Output for input x = {} should be y = {}. Computed value: {}".format(input, output, nn_single.calculate(input)))

        print('\n')
        print('XOR Gate: Multi-layer Perceptron:')
        print('--------------------------------------------------------------------------------')
        print('For y = 1, output should be >= 0.5. For y = 0, output should be < 0.5.')
        for input, output in zip(x, y):
            print("MULTI-LAYER PERCEPTRON: Output for input x = {} should be y = {}. Computed value: {}".format(input, output,
                                                                                                           nn_multi.calculate(
                                                                                                               input)))



if __name__ == "__main__":

    main()