'''
Simple RNN
'''

import numpy as np

n_samples = 3

# network will learn to output the value of one bit in the binary repr of the number given the 2 input bits
# of the input numbers at each position
# example for 2+3 = 5 which in 3-bit binary is 010 + 011 = 101, the network will be trained to add 010 + 011
# bit by bit from right to left having memorized the carry bit from the previous position. As such, the first
# inputs are 0 and 1 and output 1, then next inputs are 1 and 1 and output is 0 with 1 left to carry and finaly,
# the last inputs are 0 and 0 which is 0 but there's a 1 from the previous inputs so the final result is 1.

input_dim = 2
output_dim = 1
n_bit = 8
hidden_size = 3
learning_rate = .05

largest_input_number = pow(2, n_bit) / 2
weights_hidden = np.random.standard_normal(size=(input_dim, hidden_size))
weights_previous_hidden = np.random.standard_normal(size=(hidden_size, hidden_size))
weights_output = np.random.standard_normal(size=(hidden_size, output_dim))

def sigmoid(x): 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): x * (1 - x)

# online learning: network gets updated with each sample on the way
for i in range(n_samples):

    if (i % 100) == 0: print
    "reached {}".format(i)

    # generate 2 random numbers and their sum
    input_1, input_2 = np.random.randint(0, largest_input_number), np.random.randint(0, largest_input_number)
    true_output = input_1 + input_2

    # calculate the binaries
    input_1_binary, input_2_binary, true_output_binary = [int(x) for x in np.binary_repr(input_1, n_bit)], [int(x) for x
                                in np.binary_repr(input_2, n_bit)], [int(x) for x in np.binary_repr(true_output, n_bit)]

    previous_hidden_outputs = np.zeros(hidden_size)

    # we'll append the outputs at each layer on the way ..
    hidden_layer_output_seq = [[np.zeros(hidden_size)]]
    output_layer_output_seq = []

    # forward pass of the bit sequence through the network and accumulating the errors at each bit position
    for bit_idx in range(n_bit - 1, -1, -1):
        input_bits = [input_1_binary[bit_idx], input_2_binary[bit_idx]]

        # pass the inputs through the hidden and the output layers
        hidden_layer_outputs = sigmoid(np.dot(input_bits, weights_hidden) + np.dot(previous_hidden_outputs, weights_previous_hidden))
        output_layer_output = sigmoid(np.dot(hidden_layer_outputs, weights_output))

        # we store the outputs so that we back-propagate the errors
        hidden_layer_output_seq.append(hidden_layer_outputs)
        output_layer_output_seq.append(output_layer_output)

        previous_hidden_outputs = hidden_layer_outputs


    # back propagation through the sequence to update weights (back propagation through time)
    # you know how gradient descent works: update weights in the direction of the derivatives to optimize
    # the objective function. only that these updates should be weighted by size of errors, size of inputs and finally
    # some meta-parameter we control, called learning rate
    # general steps:
    #  1. compute error at each layer starting with the output and propagating back
    #  2. compute the derivative of the outputs at each layer and weighted is by the errors
    #  3. accumulate the error weighted derivatives at each layer for all elements in the time series sequence
    #  4. weight once more the error weighted derivatives by the size/value of the inputs
    #  5. update the weights in the direction the derivatives which were: 1. summed up for all the ]
    # elements in the sequence 2. weighted by both the size of the errors and the size of the inputs.

    # init the derivative of the outputs at the hidden layer weighted by the errors, in order to propagate back
    previous_hidden_layer_error_weighted_derivative = np.zeros(hidden_size)

    # sum of the derivative of the outputs at the corresponding layers weighted by the errors, for each pair of input bits
    sum_hidden_layer_updates = np.zeros_like(weights_hidden)
    sum_previous_hidden_layer_updates = np.zeros_like(weights_previous_hidden)
    sum_output_layer_updates = np.zeros_like(weights_output)

    # rolling back from the last bit to the first
    for bit_idx in range(n_bit):

        # take output error at this position -> size(output_dim)
        output_error = true_output_binary[bit_idx] - output_layer_output_seq[bit_idx]

        # calculate output derivative weighted by the output errors -> size(output_dim)
        output_error_weighted_derivative = np.dot(sigmoid_derivative(output_layer_output_seq[bit_idx]), output_error.T)

        # sum the output_error_weighted_derivative for each element in the sequence weighted by the size of inputs int this layer -> (hidden_size, output_dim)
        sum_output_layer_updates += np.dot(hidden_layer_output_seq[bit_idx].T, output_error_weighted_derivative)

        # calculate hidden error as coming from: 1.what was sent to the output, 2.what was sent to the next hidden layer
        #  -> (output_dim)* (hidden_size, output_dim) + (hidden_size)*(hidden_size, hidden_size) = (hidden_size)
        hidden_error = np.dot(output_error_weighted_derivative, weights_output.T) + np.dot(previous_hidden_layer_error_weighted_derivative, weights_previous_hidden)

        # calculate hidden outputs derivatives weighted by hidden errors ->(hidden_size) * (hidden_size) = (hidden_size)
        hidden_error_weighted_derivative = np.dot(sigmoid_derivative(hidden_layer_output_seq[bit_idx]), hidden_error.T)

        # sum the output_error_weighted_derivative for each element in the sequence, weighted by the size of the inputs -> (input_dim, hidden_size)
        sum_hidden_layer_updates += np.dot(np.array([input_1_binary[bit_idx], input_2_binary[bit_idx]]).T, hidden_error_weighted_derivative)

        # sum the hidden_error_weighted_derivative for each element in the sequence, weighted by the size of the inputs -> (hidden_size, hidden_size)
        sum_previous_hidden_layer_updates += np.dot(hidden_layer_output_seq[bit_idx - 1].T, hidden_error_weighted_derivative)

        # propagating the hidden layer error back to
        previous_hidden_layer_error_weighted_derivative = hidden_error_weighted_derivative

    # updating weights for this sample
    weights_hidden += sum_hidden_layer_updates * learning_rate
    weights_previous_hidden += sum_previous_hidden_layer_updates * learning_rate
    weights_output += sum_output_layer_updates * learning_rate


