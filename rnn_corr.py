'''
Simple RNN for adding 2 numbers in binary

forward forward..
backward - we want to iteratively change the weights starting from random positions
to minimize error we change the weights of each layer in the directions of the derivatives of the output  of that layer
(note that the overall error function to optimize is formed by all layer functions but as we go back we no longer care
about the functions in front and their weights..)
we want the change to be proportional to the size of the error and also the size of the input - so we weight the derivatives
by the errors deltas and inputs
when passing error delta back to previous layer - we multiply current error weighted derivative by the weights to see how much
of the erro corresponds to each of the previous layer outputs
'''

######################################### THE DATA #########################################

import numpy as np
import copy

np.random.seed(0)
from numpy import ones, zeros, zeros_like, log, clip

# the data generating params
n_samples = 10000
print_every = 1000 # how often to display training progress
n_bit = 8
largest_input_number = sum([ pow(2, i)  for i in range(n_bit) ])/ 2
print "largest number to sum up to, is {}".format(largest_input_number*2)

#### done with constants

def generate_random_sample():
    # generate 2 random numbers and their sum
    input_1, input_2 = np.random.randint(0, largest_input_number), np.random.randint(0, largest_input_number)
    true_output = input_1 + input_2

    # calculate the binaries

    input_1_binary = [int(x) for x in np.binary_repr(input_1, n_bit)]
    input_2_binary = [int(x) for x in np.binary_repr(input_2, n_bit)]
    true_output_binary = [int(x) for x in np.binary_repr(true_output, n_bit)]

    return list(reversed(input_1_binary)), list(reversed(input_2_binary)), list(reversed(true_output_binary))

############################################# THE RNN #############################################

# RNN params
input_dim = 2
output_dim = 1
recursive_size = 3
learning_rate = .1

# RNN weights
# simple RNN with one recurent hidden layer and one output layer

weights = { # hidden layer weights
           "recursive": np.random.standard_normal(size=(input_dim, recursive_size)),
           "previous_recursive": np.random.standard_normal(size=(recursive_size, recursive_size)),
           "recursive_bias": zeros((1, recursive_size)),
            # output layer weights
           "dense":np.random.standard_normal(size=(recursive_size, output_dim)),
           "dense_bias": zeros((1,output_dim)),
            # the associated metrics with this set of weights' values
            "log_loss":0
          }

# RNN Functions

# first thing first - what do we measure?
def logloss(target, predicted, eps=1e-15): return log(1-clip(predicted, eps, 1-eps))*(target-1) - log(clip(predicted, eps, 1-eps))*target
# compute the loss for a sequence of target and predicted values
def compute_loss_seq(targets, predicted):
    assert len(targets) == len(predicted)
    return np.mean([logloss(x[0], x[1]) for x in np.stack([targets, predicted], 1)])

# util math functions
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
def sigmoid_derivative(x): return x * (1 - x)


# gets an input sample and recurrent input and returns all layer outputs

def feed_forward_recursive_layer(inputs, weights):  # input_data, previous_recursive_layer_output):

    raw_outputs = np.dot(inputs["from_previous"], weights["recursive"]) + np.dot(
        inputs["from_recursive"], weights["previous_recursive"]) + weights["recursive_bias"]

    return {"raw": raw_outputs, "activation": sigmoid(raw_outputs)}


# backprop through time rnn layer
# takes: its raw output, all the errors deltas sent to its successors
# returns: the overall error delta to pass to its precedessors and the deltas to update its own weights
def backprop_recursive_layer(inputs, outputs, errors,
                             weights):  # error_to_output, error_to_next_recursive,  layer_raw_output):

    # calculate error as coming back from: 1.what was sent to the output, 2.what was sent to the next hidden layer
    error = np.dot(errors["to_output"], weights["dense"].T) + np.dot(errors["to_next_recursive"],
                                                                     weights["previous_recursive"])
    # total delta of the layer to pass further down to previous inputing layers: error_weighted_derivative of output
    total_delta = sigmoid_derivative(outputs["activation"]) * error
    # delta corresponding to input from below layer based on inputs from that layer
    input_w_delta = np.dot(inputs["from_previous"].T, total_delta)
    # delta corresponding to input from previous hidden layer based on inputs from that layer
    recursive_w_delta = np.dot(inputs["from_recursive"].T, total_delta)
    return {"total_delta": total_delta, "recursive_w_delta": recursive_w_delta, "input_w_delta": input_w_delta}


# gets an input sample and recurrent input and returns all layer outputs
def feed_forward_dense_layer(inputs, weights):
    raw_output = np.dot(inputs["from_previous"], weights["dense"]) + weights["dense_bias"]

    return {"raw": raw_output, "activation": sigmoid(raw_output)}


# gets the error delta it sent to output and the layer input and returns the delta to pass down and
# the delta to update its weights
def backprop_dense_layer(inputs, outputs, errors, weights):
    # delta at this layer
    total_delta = 1 * errors["to_output"]  # being the output dense layer, derivative = 1
    input_w_delta = np.dot(inputs["from_previous"].T, total_delta)

    return {"total_delta": total_delta, "input_w_delta": input_w_delta}


# feed forward one sample unit through all layers
def feed_forward_network(inputs, weights):
    recursive_layer_outputs = feed_forward_recursive_layer(inputs, weights)
    dense_layer_outputs = feed_forward_dense_layer({"from_previous": recursive_layer_outputs["activation"]}, weights)

    return {"from_dense": dense_layer_outputs, "from_recursive": recursive_layer_outputs}


# back prop one sample unit through all layers
# because it's recursive it takes possible deltas from successor samples feeded forward, just as the feed forward takes recursive
# outputs from previous samples
# should return/fill the updates coresponding to this sample
def back_prop_network(inputs, all_layer_outputs, target, next_sample_deltas, weights):
    inputs_dense = {"from_previous": all_layer_outputs["from_recursive"]["activation"]}
    outputs_dense = all_layer_outputs["from_dense"]
    errors_dense = {"to_output": target - all_layer_outputs["from_dense"]["activation"]}
    dense_deltas = backprop_dense_layer(inputs_dense, outputs_dense, errors_dense, weights)

    inputs_recursive = inputs
    outputs_recursive = all_layer_outputs["from_recursive"]
    errors_recursive = {"to_output": dense_deltas["total_delta"],
                        "to_next_recursive": next_sample_deltas["recursive_deltas"]["total_delta"]}
    recursive_deltas = backprop_recursive_layer(inputs_recursive, outputs_recursive, errors_recursive, weights)

    return {"dense_deltas": dense_deltas, "recursive_deltas": recursive_deltas}


# feeds forward a sequence of samples..
def feed_forward_network_sequence(inputs_seq, weights):
    all_samples_output_seq = [{"from_recursive": {"activation":zeros((1, recursive_size))}}]
    for input_unit in inputs_seq:
        input_unit["from_recursive"] = all_samples_output_seq[-1]["from_recursive"]["activation"]
        all_samples_output_seq.append(feed_forward_network(input_unit, weights))

    return all_samples_output_seq[1:]


# back propagates a sequence of samples - we don't pass delta from previous sequence here
def back_prop_network_sequence(inputs_seq, outputs_seq, target_seq, weights):
    # dense deltas are not going to be used so no init is needed
    init_recursive_deltas = {"total_delta": zeros((1, recursive_size)),
                             "recursive_w_delta": zeros_like(weights["previous_recursive"]),
                             "input_w_delta": zeros_like(weights["recursive"])}
    init_dense_deltas = {"total_delta": 0, "input_w_delta": zeros_like(weights["dense"])}
    all_deltas_seq = [{"dense_deltas": init_dense_deltas, "recursive_deltas": init_recursive_deltas}]

    for i in range(1, len(inputs_seq)+1):
        all_deltas_seq.append(
            back_prop_network(inputs_seq[-i], outputs_seq[-i], target_seq[-i], all_deltas_seq[-i], weights))

    # compute loss for the whole sequence
    weights["log_loss"] += compute_loss_seq(target_seq, [x['from_dense']['activation'][0][0] for x in outputs_seq])

    return all_deltas_seq


# update weights with a seq  of deltas coresponding to a sequence of inputs
# also compute the log loss of the previous set of weights
def update_network_weights(all_deltas_seq, weights):
    for all_deltas in all_deltas_seq:
        weights["recursive"] -= learning_rate * np.clip(all_deltas["recursive_deltas"]["input_w_delta"], -3, 3)
        weights["previous_recursive"] -= learning_rate * np.clip(all_deltas["recursive_deltas"]["recursive_w_delta"],
                                                                 -3, 3)
        weights["recursive_bias"] -= learning_rate * np.clip(all_deltas["recursive_deltas"]["total_delta"], -3, 3)
        weights["dense"] -= learning_rate * np.clip(all_deltas["dense_deltas"]["input_w_delta"], -3, 3)
        weights["dense_bias"] -= learning_rate * np.clip(all_deltas["dense_deltas"]["total_delta"], -3, 3)


def train_net(weights):
    for i in range(n_samples):

        input_1_binary, input_2_binary, target_binary = generate_random_sample()
        input_seq = [{"from_previous": np.array([x]), "from_recursive": zeros((1, recursive_size))} for x in
                     zip(input_1_binary, input_2_binary)]
        update_network_weights(
            back_prop_network_sequence(input_seq, feed_forward_network_sequence(input_seq, weights),
                                       target_binary, weights), weights)

        if i % print_every ==0:
            print "................................. sample {}".format(i)
            print float(weights["log_loss"]) / (i + 1)
            #print weights["dense"].T
            #print weights["recursive"]
            #print weights["previous_recursive"]



train_net(weights)

