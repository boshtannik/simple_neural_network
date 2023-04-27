"""
Written by boshtannik.
e-mail: boshtannik@gmail.com
Inpiration got from book "Crete own neural network" by Tariq Rashid.
The main idea was to create neural network by representing it via matricies
and their multiplication, but I decide to walk in other way, in order to
picture the whole architecture in the object-oriented way.

The main idea was to this project - was to put basement to let further reconfiguration
of the neural network architecture, i. e. mean to:
    * Being able to link new neurons into the existing neural network.
    * Cut their pieces like connections.
    * Connect other neural networks to newly created neurons of existing neural network.
    * Having fun.
    * etc..

What else to be expected:
    * Being able to save neurons connections weights into file.
    * Restore previously trained neural network connection weights from file.
    * Run neural network predict method from command line.
    * Support of biasses to be added into hidden layers.
"""


from math import exp
from typing import List
from random import randint, choice


class Connection:
    def __init__(self, from_neuron, to_neuron, weight):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron

        self.weight = weight

        to_neuron.input_connections.append(self)
        from_neuron.output_connections.append(self)

    def __repr__(self):
        return f'Connection({self.from_neuron}, {self.to_neuron}, {self.weight})'

    def get_weighted_value(self):
        return self.from_neuron.output * self.weight

    def forward(self):
        self.to_neuron.input += self.get_weighted_value()

    def back_propagate(self):
        self.from_neuron.error += self.to_neuron.error * self.weight

    def adjust_weight(self, learning_rate: float = 1.0):
        self.weight += (learning_rate * self.from_neuron.output * self.to_neuron.get_error_derivative())
        

class Neuron:
    def __init__(self):
        self.error = 0
        self.input = 0
        self.output = 0
        self.output_connections = []
        self.input_connections = []

    def activation_function(self, x):
        return 1 / (1 + exp(-x))

    def derivative_activation_function(self, x):
        return x * (1 - x)

    def activate(self):
        self.output = self.activation_function(self.input)

    def get_error(self, expected_value: float) -> float:
        self.error = expected_value - self.output
        return self.error

    def get_error_derivative(self) -> float:
        return self.error * self.derivative_activation_function(self.output)


class Perceptron:
    def __init__(self, layers: List[int]):
        """
        Param layers - is list of counts of neurons in each layer
        """
        self.layers = []
        for layer in layers:
            self.layers.append([Neuron() for _ in range(layer)])

        for i in range(len(self.layers) - 1):
            for from_neuron in self.layers[i]:
                for to_neuron in self.layers[i + 1]:
                    Connection(from_neuron, to_neuron, randint(-100, 100) * 0.01)

        # hidden layers are all except input and output
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def predict(self, inputs: List[float]) -> List[float]:
        if len(inputs) != len(self.input_layer):
            raise ValueError('Number of inputs must be equal to number of input neurons')

        # Set the output of the input layer to the input values
        for i in range(len(inputs)):
            self.input_layer[i].output = inputs[i]

        # Propagate the values through the network
        for i in range(len(self.layers) - 1):
            # current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            for next_neuron in next_layer:
                next_neuron.input = 0
                for connection in next_neuron.input_connections:
                    connection.forward()
                next_neuron.activate()

        # Return the output values of the output layer
        return [neuron.output for neuron in self.output_layer]

    def get_errors(self, expected_values: List[float]) -> List[float]:
        if len(expected_values) != len(self.output_layer):
            raise ValueError('Number of expected values must be equal to number of output neurons')

        errors = []
        for i in range(len(expected_values)):
            errors.append(self.output_layer[i].get_error(expected_values[i]))
        return errors
    
    def train(self, inputs: List[float], expected_values: List[float], learning_rate: float):
        if len(inputs) != len(self.input_layer):
            raise ValueError('Number of inputs must be equal to size of input layer')

        if len(expected_values) != len(self.output_layer):
            raise ValueError('Number of expected_values must be equal to size of output layer')
        
        # Forward propagation
        self.predict(inputs)
        
        # Calculate the error for each output neuron
        errors = self.get_errors(expected_values)
        total_error = sum(errors)

        # 1 Clean all neuron errors
        for layer in self.layers:
            for neuron in layer:
                neuron.error = 0

        # 2 Set errors to last layer
        for neuron in self.output_layer:
            neuron.error = errors.pop(0)

        # 3 Iterate trough layers from last to previous ones, in order to:
        #   1 - propagate error back.
        #   2 - adjust weights accordingly to propagated error.
        # For every income connection of neuron - call:
        #       1 back_propagate() -> Should: get error of current neuron,
        #                             multiply it by weight of connection, and add
        #                             result value to error of previous neuron
        #       2 adjust_weight()  -> Should: add to weight result of (learning_rate * connection.from_neuron.output * connection.to_neuron.get_error_derivative)

        # Backpropagation
        for layer in reversed(self.layers):
            for neuron in layer:
                for input_connection in neuron.input_connections:
                    input_connection.back_propagate()
                    input_connection.adjust_weight(learning_rate=learning_rate)
        
        # Print the average error for this epoch
        print(f'Average error: {total_error/len(inputs)}')


def test_neural_network():
    """
    This is almost default hello world test case for testing
    newly baked neural network.
    This test simulates, how the neural network did understand
    the logic behind XOR logic module.

    The XOR logic truth table shall look like this
    ------------------------------
    | input 1 | input 2 | output |
    ------------------------------
    |    0    |    0    |    0   |
    ------------------------------
    |    0    |    1    |    1   |
    ------------------------------
    |    1    |    0    |    1   |
    ------------------------------
    |    1    |    1    |    0   |
    ------------------------------
    """
    # Define the neural network architecture

    # Smallest configuration for this task that i found is: 2, 3, 2, 1. Works in ~30% Cases. Often shits itself.
    nn = Perceptron(layers=[2, 5, 6, 2, 1])

    # Set data for training.
    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]

    expected_values = [
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ]

    # Prepare data in order (<Data to be feed in>, <Expected data to be received>)
    train_data = list(zip(inputs, expected_values))

    # Train it 100000 times.
    for _ in range(100000):
        input, expected_val = choice(train_data)
        nn.train(input, expected_val, learning_rate=0.1)

    # Test the neural network
    got_1_prediction, = nn.predict(inputs=inputs[0])
    got_1_prediction *= 100  # Convert to percents
    test_1_passed = got_1_prediction < 3  # Check if confidence in activation is less than 3%
    print(f"Test with input data: {inputs[0]} is {'' if test_1_passed else 'not '}passed. with result: {got_1_prediction:.2f}% confidence.")

    got_2_prediction, = nn.predict(inputs=inputs[1])
    got_2_prediction *= 100  # Convert to percents
    test_2_passed = got_2_prediction > 97  # Check if confidence in activation is less than 3%
    print(f"Test with input data: {inputs[1]} is {'' if test_2_passed else 'not '}passed. with result: {got_2_prediction:.2f}% confidence.")

    got_3_prediction, = nn.predict(inputs=inputs[2])
    got_3_prediction *= 100  # Convert to percents
    test_3_passed = got_3_prediction > 97  # Check if confidence in activation is less than 3%
    print(f"Test with input data: {inputs[2]} is {'' if test_3_passed else 'not '}passed. with result: {got_3_prediction:.2f}% confidence.")

    got_4_prediction, = nn.predict(inputs=inputs[3])
    got_4_prediction *= 100  # Convert to percents
    test_4_passed = got_4_prediction < 3  # Check if confidence in activation is less than 3%
    print(f"Test with input data: {inputs[3]} is {'' if test_4_passed else 'not '}passed. with result: {got_4_prediction:.2f}% confidence.")

    all_passed = all((test_1_passed, test_2_passed, test_3_passed, test_4_passed))
    print(f"Finally. The neural network did{('' if all_passed else ' not')} pass the test", )


if __name__ == '__main__':
    test_neural_network()
