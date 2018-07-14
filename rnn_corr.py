
'''
Simple RNN for adding 2 numbers in binary
'''

import numpy as np
import copy


n_samples = 150#0#0
np.random.seed(0)


input_dim = 2
output_dim = 1
n_bit = 8
hidden_size = 16
learning_rate = .01

largest_input_number = pow(2, n_bit) / 2
weights_hidden = np.random.standard_normal(size=(input_dim, hidden_size))
weights_previous_hidden = np.random.standard_normal(size=(hidden_size, hidden_size))
weights_output = np.random.standard_normal(size=(hidden_size, output_dim))

def sigmoid(x): return (1 / (1 + np.exp(-x)))
def sigmoid_derivative(x): return x * (1 - x)

batch_error = 0

# online learning: network gets updated with each sample on the way
for i in range(n_samples):

    # generate 2 random numbers and their sum
    input_1, input_2 = np.random.randint(0, largest_input_number), np.random.randint(0, largest_input_number)
    true_output = input_1 + input_2

    # calculate the binaries
    input_1_binary = [int(x) for x in np.binary_repr(input_1, n_bit)]
    input_2_binary = [int(x) for x in np.binary_repr (input_2, n_bit)]
    true_output_binary = [int(x) for x in np.binary_repr(true_output, n_bit)]


    # we'll append the outputs at each layer on the way..
    hidden_layer_output_seq = []
    hidden_layer_output_seq.append(np.zeros((1, hidden_size)))
    output_layer_output_seq = []
    previous_output_layer_inputs = np.zeros((1, hidden_size))

    # forward pass of the bit sequence through the network and accumulating the errors at each bit position
    for bit_idx in range(n_bit - 1, -1, -1):
        input_bits = np.array([[input_1_binary[bit_idx], input_2_binary[bit_idx]]])
        hidden_layer_outputs = np.dot(input_bits, weights_hidden) + np.dot(previous_output_layer_inputs,weights_previous_hidden)

        output_layer_inputs = sigmoid(hidden_layer_outputs)
        output_layer_output = np.dot(output_layer_inputs, weights_output)

        rnn_response = sigmoid(output_layer_output)

        hidden_layer_output_seq.append(copy.copy(hidden_layer_outputs))
        output_layer_output_seq.append(copy.copy(output_layer_output))

        previous_output_layer_inputs = output_layer_inputs

    previous_hidden_layer_error_weighted_derivative = np.zeros((1, hidden_size))
    # append one more zero array for going backwards

    # sum of the derivative of the outputs at the corresponding layers weighted by the errors, for each pair of input bits
    sum_hidden_layer_updates = np.zeros_like(weights_hidden)
    sum_previous_hidden_layer_updates = np.zeros_like(weights_previous_hidden)
    sum_output_layer_updates = np.zeros_like(weights_output)

    # rolling back from the last bit to the first
    hidden_layer_output_seq.reverse()
    output_layer_output_seq.reverse()
    #print output_layer_output_seq


    for bit_idx in range(n_bit):
        # take output error at this position -> size(output_dim)
        output_error = np.array([true_output_binary[bit_idx]]) - sigmoid(output_layer_output_seq[bit_idx])
        #print output_layer_output_seq[bit_idx]
        # calculate output derivative weighted by the output errors -> size(output_dim)
        # output_error_weighted_derivative = sigmoid_derivative(output_layer_output_seq[bit_idx]) * output_error

        #print output_error_weighted_derivative
        # sum the output_error_weighted_derivative for each element in the sequence weighted by the size of inputs int this layer -> (hidden_size, output_dim)
        sum_output_layer_updates += np.dot(sigmoid(hidden_layer_output_seq[bit_idx]).T, output_error)

        # calculate hidden error as coming from: 1.what was sent to the output, 2.what was sent to the next hidden layer
        #  -> (output_dim)* (hidden_size, output_dim) + (hidden_size)*(hidden_size, hidden_size) = (hidden_size)

        hidden_error = np.dot(output_error, weights_output.T) + np.dot(previous_hidden_layer_error_weighted_derivative, weights_previous_hidden)

        # calculate hidden outputs derivatives weighted by hidden errors ->(hidden_size) * (hidden_size) = (hidden_size)
        # print hidden_layer_output_seq[bit_idx].T.shape
        # print hidden_error.shape

        hidden_error_weighted_derivative = sigmoid_derivative(hidden_layer_output_seq[bit_idx]) * hidden_error
        # print hidden_error_weighted_derivative.shape

        # sum the output_error_weighted_derivative for each element in the sequence, weighted by the size of the inputs -> (input_dim, hidden_size)
        sum_hidden_layer_updates += np.dot(np.array([[input_1_binary[bit_idx], input_2_binary[bit_idx]]]).T,
                                           hidden_error_weighted_derivative)

        # sum the hidden_error_weighted_derivative for each element in the sequence, weighted by the size of the inputs -> (hidden_size, hidden_size)
        sum_previous_hidden_layer_updates += np.dot(hidden_layer_output_seq[bit_idx - 1].T,
                                                    hidden_error_weighted_derivative)

        # propagating the hidden layer error back to
        previous_hidden_layer_error_weighted_derivative = hidden_error_weighted_derivative

        # just accumulating error for printing
        batch_error += abs(output_error[0])

    # updating weights for this sample
    #weights_hidden += (sum_hidden_layer_updates * learning_rate)
    #weights_previous_hidden += (sum_previous_hidden_layer_updates * learning_rate)
    weights_output += (sum_output_layer_updates * learning_rate)
    print weights_hidden

'''
    errors = np.array(true_output_binary) - np.array([sigmoid(x.tolist()[0][0]) for x in output_layer_output_seq])
    batch_error += sum([abs(x) for x in errors]) / n_bit

    if (i % 1000) == 0:
        print
        100 * '#' + " sample {} ".format(i)
        print
        " Training sample: {0} + {1} = {2}".format(input_1, input_2, true_output)
        # print " Binary version: {0} + {1} = {2}".format(input_1_binary, input_2_binary, true_output_binary)
        result = [sigmoid(x.tolist()[0][0]) for x in output_layer_output_seq]
        print
        " Result is {}".format(sum([pow(2, n_bit - i - 1) * round(result[i]) for i in range(n_bit)]))
        # print result

        print
        " Average binarry error for this batch is {}".format(batch_error / 8000)
        batch_error = 0

'''