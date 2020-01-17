import numpy as np

class Neuron:

    # Initialize
    def __init__(self, num_inputs, learning_rate, activation, weights, bias):

        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights = weights
        self.bias = bias

    # Activation function. Choices include
    #   - Sigmoid
    #   - ReLu
    #   - Linear
    #   - Step
    def activate(self, input):

        if self.activation == "sigmoid":
            return (1/(1+np.exp(-input)))

        elif self.activation == "relu":
            return np.max(0, input)

        elif self.activation == "linear":
            return input

        elif self.activation == "step":
            return (input >= 0).astype(int)

    def calculate(self, inputs):

        assert(len(inputs) == len(self.weights))

        return self.activate(np.dot(inputs, self.weights) + self.bias)


class FullyConnectedLayer:

    # Initialize
    def __init__(self, num_neurons, activation, num_inputs, learning_rate):

        self.num_neurons = num_neurons
        self.activation = activation
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate

        self.weights = np.random.randn(num_neurons*num_inputs)
        self.bias = np.random.randn(num_neurons)

    def calculate(self, inputs):

        # Initialize the outputs of the layer and an empty list to store Neuron objects
        layer_out = [None]*self.num_neurons
        neurons = [None]*self.num_neurons

        # Iterate through each neuron (inefficient...)
        for neuron in range(0, self.num_neurons):

            # Get indices for weight array at this iteration's neuron
            n_indices = np.linspace(neuron*self.num_inputs, neuron*self.num_inputs+(self.num_inputs-1), self.num_inputs).astype(int)

            # Create neuron using the Neuron class
            neurons[neuron] = Neuron(self.num_inputs, self.learning_rate, self.activation, self.weights[n_indices], self.bias[neuron])

            # Compute the output of this neuron
            layer_out[neuron] = neurons[neuron].calculate(inputs)

        return np.asarray(layer_out)


class NeuralNetwork:

    def __init__(self, num_hidden_layers, num_neurons, activations, num_inputs, loss_function, learning_rate):

        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.activations = activations
        self.num_inputs = num_inputs
        self.loss_function = loss_function
        self.learning_rate = learning_rate