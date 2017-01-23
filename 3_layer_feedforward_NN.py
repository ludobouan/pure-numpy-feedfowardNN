import numpy as np


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator
        np.random.seed(1)
        self.weights = {}
        self.num_layers = 1
        self.adjustments = {}
        self.biases = {}

    def add_layer(self, shape):
        self.weights[self.num_layers] = np.vstack((2 * np.random.random(shape) - 1, 2 * np.random.random((1, shape[1])) - 1))
        self.adjustments[self.num_layers] = np.zeros(shape)
        self.num_layers += 1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, data):
        for layer in range(1, self.num_layers+1):
            data = np.dot(data, self.weights[layer-1][:, :-1]) + self.weights[layer-1][:, -1] # + self.biases[layer]
            data = self.__sigmoid(data)
        return data

    def __forward_propagate(self, data):
        activation_values = {}
        activation_values[1] = data
        for layer in range(2, self.num_layers+1):
            data = np.dot(data.T, self.weights[layer-1][:-1, :]) + self.weights[layer-1][-1, :].T # + self.biases[layer]
            data = self.__sigmoid(data).T
            activation_values[layer] = data
        return activation_values

    def simple_error(self, outputs, targets):
        return targets - outputs

    def sum_squared_error(self):
        return 0.5 * np.mean(np.sum(np.power(outputs - targets, 2), axis=1))

    def __back_propagate(self, output, target):
        deltas = {}
        # Output Layer
        deltas[self.num_layers] = output[self.num_layers] - target

        # Hidden Layers
        for layer in reversed(range(2, self.num_layers)):  # All layers except input/output
            a_val = output[layer]
            weights = self.weights[layer][:-1, :]
            prev_deltas = deltas[layer+1]
            deltas[layer] = np.multiply(np.dot(weights, prev_deltas), self.__sigmoid_derivative(a_val))

        for layer in range(1, self.num_layers):
            self.adjustments[layer] += np.dot(deltas[layer+1], output[layer].T).T

    def __gradient_descente(self, batch_size, learning_rate):
        for layer in range(1, self.num_layers):
            partial_d = (1/batch_size) * self.adjustments[layer]
            self.weights[layer][:-1, :] += learning_rate * -partial_d
            self.weights[layer][-1, :] += 0.001 * -partial_d[-1, :]


    def train(self, inputs, targets, num_epochs, learning_rate=0.1):
        for iteration in range(num_epochs):
            for i in range(len(inputs)):
                x = inputs[i]
                y = targets[i]
                # Pass the training set through our neural network
                output = self.__forward_propagate(x)

                # Calculate the error
                error = self.simple_error(output[self.num_layers], y)
                print('error = ', error)

                # Calculate Adjustements
                self.__back_propagate(output, y)
            if iteration%10 == 0:
                self.__gradient_descente(40, learning_rate)



if __name__ == "__main__":

    # Create instance of a neural network
    nn = NeuralNetwork()

    # Add Layers (Input layer is created by default)
    nn.add_layer((2, 4))
    nn.add_layer((4, 1))

    # XOR function
    training_data = np.asarray([[0, 0], [0, 1], [1, 0], [0, 0]]).reshape(4, 2, 1)
    training_labels = np.asarray([[0], [1], [1], [0]])

    # XOR function
    testing_data = np.asarray([[0, 0], [0, 1], [1, 0], [0, 0]]).reshape(4, 2, 1)
    training_labels = np.asarray([[0], [1], [1], [0]])

    nn.train(training_data, training_labels, 10000)
    # nn.predict(testing_data)
