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
    #   - Sigmoid (Logistic)
    #   - ReLu
    #   - Linear
    def activate(self, input):

        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-input))

        elif self.activation == "relu":
            return np.max(0, input)

        elif self.activation == "linear":
            return input

    def activate_derivative(self, input):

        if self.activation == "sigmoid":
            return self.activate(input) * (1 - self.activate(input))

        elif self.activation == "relu":
            return np.asarray(input > 0).astype(int)  # Returns "1" if input is greater than zero, "0" otherwise
                                                      # Derivative of ReLU function is the step function
        elif self.activation == "linear":
            return 1

    def calculate(self, inputs):

        assert(len(inputs) == len(self.weights))

        net = np.dot(inputs, self.weights) + self.bias
        out = self.activate(net)

        self.d_out_d_net = self.activate_derivative(net)

        return out


class FullyConnectedLayer:

    # Initialize
    def __init__(self, num_neurons, activation, num_inputs, learning_rate):

        self.num_neurons = num_neurons
        self.activation = activation
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate

        self.weights = np.random.randn(num_neurons*num_inputs)
        self.bias = np.random.randn(num_neurons)

        # Initialize  an empty list to store Neuron objects
        self.neurons = [None] * self.num_neurons

        # Iterate through each neuron (inefficient...)
        for neuron in range(0, self.num_neurons):

            # Get indices for weight array at this iteration's neuron
            n_indices = np.linspace(neuron * self.num_inputs, neuron * self.num_inputs + (self.num_inputs - 1),
                                    self.num_inputs).astype(int)

            # Create neuron using the Neuron class
            self.neurons[neuron] = Neuron(self.num_inputs, self.learning_rate, self.activation, self.weights[n_indices],
                                          self.bias[neuron])

    def calculate(self, inputs):

        # A variable to store the output of the neurons in the current layer (for use in backprop)
        self.layer_out = np.zeros(self.num_neurons)

        # Iterate through each neuron (inefficient...)
        for neuron in range(0, self.num_neurons):

            # Compute the output of this neuron
            self.layer_out[neuron] = self.neurons[neuron].calculate(inputs)

        return self.layer_out


class NeuralNetwork:

    def __init__(self, parameters):

        self.num_hidden_layers = parameters['num hidden layers']

        if parameters['num hidden layers'] == 0:
            self.num_neurons = parameters['num inputs']

        else:
            self.num_neurons = parameters['num neurons']

        self.num_outputs = parameters['num outputs']

        # Make sure the number of chosen activation functions is equal to the number of hidden layers + 1 (for the output layer)
        assert(len(parameters['activations']) == (parameters['num hidden layers'] + 1))

        self.activations = parameters['activations']
        self.num_inputs = parameters['num inputs']
        self.loss_function = parameters['loss function']
        self.learning_rate = parameters['learning rate']

        # Initialize layers and weights
        self.layers = [None] * (self.num_hidden_layers + 1)

        # Input and Hidden layers
        for l in range(0, self.num_hidden_layers):

            # Input
            if l == 0:
                self.layers[l] = FullyConnectedLayer(self.num_neurons, self.activations[l], self.num_inputs, self.learning_rate)

            # Hidden
            else:
                self.layers[l] = FullyConnectedLayer(self.num_neurons, self.activations[l], self.num_neurons, self.learning_rate)

        # Output layer

        self.layers[-1] = FullyConnectedLayer(num_neurons = parameters['num outputs'], activation = self.activations[-1],
                                         num_inputs = self.num_neurons, learning_rate = self.learning_rate)

    # Feedforward
    def calculate(self, input):

        cur_inputs = input

        for layer in range(0, len(self.layers)):
            cur_outputs = self.layers[layer].calculate(cur_inputs)
            cur_inputs = cur_outputs

        return cur_outputs

    # Loss function calculation
    def calculateloss(self, desired_output, actual_output):

        if self.loss_function == "squared error":
            self.loss = (1 / 2) * np.sum((desired_output - actual_output)**2)
            return self.loss

        elif self.loss_function == "binary cross entropy":
            self.loss = np.sum(actual_output * np.log(desired_output) + (1 - actual_output) * np.log(1 - desired_output))
            return self.loss

    def loss_derivative(self, desired_output, actual_output):

        if self.loss_function == "squared error":
            self.loss_deriv = -np.sum(desired_output - actual_output)
            return self.loss_deriv

        elif self.loss_function == "binary cross entropy":
            self.loss_deriv = np.sum(-(actual_output / desired_output) + ((1 - actual_output)/(1 - desired_output)))

    def update_weights(self, deltas, input):

        num_total_layers = len(self.layers)

        # Iterate through each layer
        for layer in range(0, num_total_layers):

            # Iterate through each neuron to update its input weights
            for neur in range(0, len(self.layers[layer].neurons)):

                if layer != (num_total_layers - 1):
                    delta = deltas[neur, layer]

                else:
                    delta = self.out_deltas[neur]

                if layer == 0:

                    self.layers[layer].neurons[neur].weights -= self.learning_rate * (delta * input)  # Weights
                    self.layers[layer].neurons[neur].bias -= self.learning_rate * (delta * 1)

                else:
                    outs = self.layers[layer - 1].layer_out

                    self.layers[layer].neurons[neur].weights -= self.learning_rate * (delta * outs)
                    self.layers[layer].neurons[neur].weights -= self.learning_rate * (delta * 1)

    # One iteration of gradient descent
    def train(self, inputs, desired_outputs, epochs):

        self.inputs = inputs
        N = inputs.shape[0]

        self.loss_epoch = np.zeros(epochs)

        for iter in range(0, epochs):

            self.losses = [None] * N
            out = np.zeros(desired_outputs.shape)

            # Compute each input through the network to get overall loss and update weights
            for i in range(0, len(inputs)):

                out[i] = self.calculate(inputs[i])
                # print('Feedforward output: {}'.format(out))

                self.calculateloss(desired_outputs[i], out[i])
                self.losses[i] = self.loss
                # print('Loss before update: {}'.format(losses[i]))

                self.out_deltas = np.zeros(len(self.layers[-1].neurons))
                hidden_deltas = np.zeros([self.num_neurons, self.num_hidden_layers])

                # Output layer deltas
                for out_neuron in range(len(self.layers[-1].neurons)):

                    if len(desired_outputs.shape) > 1:  # If we have more than one input/one output in the training set
                        self.out_deltas[out_neuron] = self.loss_derivative(desired_outputs[:, out_neuron], out[:, out_neuron]) * self.layers[-1].neurons[out_neuron].d_out_d_net
                    else:
                        self.out_deltas[out_neuron] = self.loss_derivative(desired_outputs, out) * self.layers[-1].neurons[out_neuron].d_out_d_net

                # Hidden layer deltas
                for hidden_layer in range(self.num_hidden_layers-1, -1, -1):  # Work backwards from output layer

                    for hidden_neuron in range(0, self.num_neurons):  # Iterate through each neuron in the layer, starting from the "top"

                        # Derivative of activation function for this neuron
                        phi_prime = self.layers[hidden_layer].neurons[hidden_neuron].d_out_d_net

                        if hidden_layer == (self.num_hidden_layers - 1):  # If we need the weights from the output layer, ...

                            out_weights = np.zeros(len(self.layers[-1].neurons))

                            for out_neuron in range(len(self.layers[-1].neurons)):
                                # Weights leaving each neuron in the final hidden layer are the "hidden_neuron"-th weight
                                # entering each output neuron. For example, for 3 hidden neurons and 2 output units, the "top"
                                # weight entering each of the output neurons will be used for the deltas in the first hidden unit.
                                # For the second hidden neuron, the "middle" weights entering each output neuron will be used
                                # for the deltas in the second hidden neuron, and so on.
                                out_weights[out_neuron] = self.layers[-1].neurons[out_neuron].weights[hidden_neuron]


                            hidden_deltas[hidden_neuron, hidden_layer] = phi_prime * np.dot(self.out_deltas, out_weights)

                        else:

                            prev_layer_deltas = hidden_deltas[:, hidden_layer + 1]
                            prev_layer_weights = np.zeros(len(self.layers[hidden_layer + 1].neurons))

                            for w in range(0, len(self.layers[hidden_layer + 1].neurons)):
                                prev_layer_weights[w] = self.layers[hidden_layer + 1].neurons[w].weights[hidden_neuron]

                            hidden_deltas[hidden_neuron, hidden_layer] = phi_prime * np.dot(prev_layer_weights, prev_layer_deltas)

                self.update_weights(hidden_deltas, inputs[i])

            print(self.losses)
            self.loss_epoch[iter] = np.mean(self.losses)
