import numpy as np
from numpy import ones, zeros, zeros_like, log, clip
######################################### THE DATA #########################################
np.random.seed(0)
n_samples, print_every, n_bit = 10000, 1000, 8,
largest_input_number = pow(2,n_bit-1) - 1
def generate_random_sample():
    input_1, input_2 = np.random.randint(0, largest_input_number), np.random.randint(0, largest_input_number)
    input_1_binary = [int(x) for x in np.binary_repr(input_1, n_bit)]
    input_2_binary = [int(x) for x in np.binary_repr(input_2, n_bit)]
    target_binary = [int(x) for x in np.binary_repr(input_1 + input_2, n_bit)]
    return list(reversed(input_1_binary)), list(reversed(input_2_binary)), list(reversed(target_binary)), input_1, input_2

############################################# THE RNN #############################################
input_dim, output_dim, recursive_size, learning_rate = 2, 1, 16, .05
# RNN weights : simple RNN with one recurrent hidden layer and one output layer
weights = { # layer for computing raw lstm values - to be filtered using the input gate
            "input_to_lstm_unfiltered": np.random.standard_normal(size=(input_dim, recursive_size)),
            "previous_lstm_output_to_lstm_unfiltered": np.random.standard_normal(size=(recursive_size, recursive_size)),
            "lstm_unfiltered_bias": zeros((1, recursive_size)),

            "input_to_input_gate": np.random.standard_normal(size=(input_dim, recursive_size)),
            "previous_lstm_output_to_input_gate": np.random.standard_normal(size=(recursive_size, recursive_size)),
            "input_gate_bias": zeros((1, recursive_size)),

            "input_to_forget_gate": np.random.standard_normal(size=(input_dim, recursive_size)),
            "previous_lstm_output_to_forget_gate": np.random.standard_normal(size=(recursive_size, recursive_size)),
            "forget_gate_bias": zeros((1, recursive_size)),

            "input_to_lstm_output_gate": np.random.standard_normal(size=(input_dim, recursive_size)),
            "previous_lstm_output_to_output_gate": np.random.standard_normal(size=(recursive_size, recursive_size)),
            "output_gate_bias": zeros((1, recursive_size)),

            # output layer weights
            "dense": np.random.standard_normal(size=(recursive_size, output_dim)),
            "dense_bias": zeros((1,output_dim)),
            # the associated metrics with this set of weights' values
            "log_loss":0 }

# first thing first - what do we measure?
def logloss(target, predicted, eps=1e-15):
    return log( 1 -clip(predicted, eps, 1- eps)) * (target - 1) - log(clip(predicted, eps, 1 - eps)) * target

# compute the loss for a sequence of target and predicted values
def compute_loss_seq(targets, predicted):
    return sum([logloss(x[0], x[1]) for x in np.stack([targets, predicted], 1)])

# util math functions
def sigmoid(x): return (1 / (1 + np.exp(-x)))
def sigmoid_derivative(x): return x * (1 - x)
def tanh(x): return (np.exp(x) - np.exp(-x))  / (np.exp(x) + np.exp(-x))
def tanh_derivative(x): return 1-tanh(x)^2

def feed_forward_to_lstm_unfiltered(inputs, weights):
    return {"activation": tanh(np.dot(inputs["from_input"], weights["input_to_lstm_unfiltered"]) +
                            np.dot(inputs["from_previous_lstm"],weights["previous_lstm_output_to_lstm_unfiltered"]) +
                            weights["lstm_unfiltered_bias"])}

def feed_forward_to_input_gate(inputs, weights):
    return {"activation": sigmoid(np.dot(inputs["from_input"], weights["input_to_input_gate"]) +
                                np.dot(inputs["from_previous_lstm"], weights["previous_lstm_output_to_input_gate"]) +
                                weights["input_gate_bias"])}

def feed_forward_to_forget_gate(inputs, weights):
    return {"activation": sigmoid(np.dot(inputs["from_input"], weights["input_to_forget_gate"]) +
                                np.dot(inputs["from_previous_lstm"], weights["previous_lstm_output_to_forget_gate"]) +
                                weights["forget_gate_bias"])}

# computing the actual lstm cell value by applying input gate to unfiltered input and forget gate to previous lstm cell value
def feed_forward_through_lstm_filter(inputs, weights):

    unfiltered_values = feed_forward_to_lstm_unfiltered(inputs, weights)["activation"]
    input_gate = feed_forward_to_input_gate(inputs, weights)["activation"]
    forget_gate = feed_forward_to_forget_gate(inputs, weights)["activation"]

    return tanh(unfiltered_values*input_gate + forget_gate * inputs["from_previous_lstm_filter"])

def feed_forward_to_output_gate(inputs, weights):
    return {"activation": sigmoid(np.dot(inputs["from_input"], weights["input_to_output_gate"]) +
                                np.dot(inputs["from_previous_lstm"], weights["previous_lstm_output_to_output_gate"]) +
                                weights["output_gate_bias"])}

def feed_forward_through_output_gate(inputs, weights):
    filtered_input = feed_forward_through_lstm_filter(inputs, weights)
    output_gate = feed_forward_to_output_gate(inputs, weights)
    return filtered_input*output_gate["activation"]

# gets an input sample and recurrent input and returns all layer outputs
def feed_forward_dense_layer(inputs, weights):
    return {"activation": sigmoid(np.dot(inputs["from_lstm"], weights["dense"]) + weights["dense_bias"])}

# feed forward one sample unit through all layers
def feed_forward_network(inputs, weights):

    unfiltered_values = feed_forward_to_lstm_unfiltered(inputs, weights)
    input_gate = feed_forward_to_input_gate(inputs, weights)
    forget_gate = feed_forward_to_forget_gate(inputs, weights)
    filtered_values = feed_forward_through_lstm_filter(inputs, weights)
    output_gate = feed_forward_to_output_gate(inputs, weights)
    lstm_output = feed_forward_through_output_gate(inputs, weights)
    dense = feed_forward_dense_layer(inputs, weights)
    return {"unfiltered_values": unfiltered_values,
            "input_gate": input_gate,
            "forget_gate" : forget_gate,
            "filtered_values" : filtered_values,
            "output_gate" : output_gate,
            "lstm" : lstm_output,
            "dense": dense}

# feeds forward a sequence of samples..
def feed_forward_sequence(inputs_seq, weights):

    all_samples_output_seq = [{"filtered_values": {"activation": zeros((1, recursive_size))},
                               "lstm_output": {"activation": zeros((1, recursive_size))}}]
    for input_unit in inputs_seq:
        input_unit["from_previous_lstm"] = all_samples_output_seq[-1]["lstm_output"]["activation"]
        input_unit["from_previous_lstm_filter"] = all_samples_output_seq[-1]["filtered_values"]["activation"]
        all_samples_output_seq.append(feed_forward_network(input_unit, weights))
    return all_samples_output_seq[1:]

# back propagates a sequence of samples - we don't pass delta from previous sequence here
# recurrences:
# lstm final output sends errors to next unfiltered values and input, forget and output gates - then deltas should go back
# lstm filtered values send errors to next lstm filter - so these deltas should be also passed back
def back_prop_sequence(inputs_seq, outputs_seq, target_seq, weights):

    init_generic_deltas2 = {"total_delta": zeros((1, recursive_size)),
                             "recursive_delta": zeros_like(weights["previous_recursive"]),
                             "input_delta": zeros_like(weights["recursive"])}

    init_generic_deltas = {"total_delta": zeros((1, recursive_size))}#, "input_w_delta": zeros_like(weights["dense"])}

    all_deltas_seq = [{"unfiltered": init_generic_deltas,
                       "input_gate": init_generic_deltas,
                       "forget_gate": init_generic_deltas,
                       "filter": init_generic_deltas,
                       "output_gate": init_generic_deltas,}]

    for i in range(1, len(inputs_seq) + 1):
        deltas = back_prop_sample(inputs_seq[-i], outputs_seq[-i], target_seq[-i], all_deltas_seq[-1], weights)
        all_deltas_seq.append(deltas.copy())
    weights["log_loss"] += compute_loss_seq(target_seq, [x['from_dense']['activation'][0][0] for x in outputs_seq])
    return all_deltas_seq[1:]

# back prop one sample unit through all layers :because it's recursive it takes possible deltas from successor
# samples fed forward, just as the feed forward takes recursive outputs from previous samples
def back_prop_sample(inputs, all_layer_outputs, target, next_sample_deltas, weights):
    inputs_dense = {"from_lstm": all_layer_outputs["lstm"]["activation"]}
    outputs_dense = all_layer_outputs["dense"]
    errors_dense = {"to_output": target - all_layer_outputs["dense"]["activation"]}
    dense_deltas = backprop_dense_layer(inputs_dense, outputs_dense, errors_dense, weights)

    inputs_lstm = {"from_input_gate": all_layer_outputs["input_gate"]["activation"],
                   "from_filter": all_layer_outputs["filter"]["activation"]}
    errors_lstm = {"to_next_unfiltered": next_sample_deltas["unfiltered"],
                   "to_next_input_gate": next_sample_deltas["input_gate"],
                   "to_next_forget_gate": next_sample_deltas["forget_gate"],
                   "to_next_output_gate": next_sample_deltas["output_gate"],
                   "to_dense": dense_deltas}
    # doesnt need the output as it doesnt have activation to compute the derivative for - it will be 1
    # instead because it's a multiplication it will have to distinct partial derivatives with respect to each factor
    lstm_deltas = backprop_through_output_gate(inputs_lstm,errors_lstm)

    inputs_output_gate = inputs_input_gate = inputs_forget_gate = inputs_unfiltered = \
        {"from_lstm": all_layer_outputs["lstm"]["activation"], "from_input": all_layer_outputs["input"]["activation"]}
    outputs_output_gate = all_layer_outputs["output_gate"]
    errors_output_gate = {"to_lstm_output": lstm_deltas["filter_delta"]}
    output_gate_delta = backprop_output_gate(inputs_output_gate, outputs_output_gate, errors_output_gate, weights)

    inputs_filter = {"from_input": all_layer_outputs["input"]["activation"],
                     "from_input_gate": all_layer_outputs["input_gate"]["activation"],
                     "from_forget_gate": all_layer_outputs["forget_gate"]["activation"],
                     "from_previous_filter": inputs["filter"]}
    #//outputs_filter = all_layer_outputs["filter"]
    errors_filter = {"to_lstm_output": lstm_deltas["output_gate_delta"],
                     "to_next_filter": next_sample_deltas["filter"]}
    # it contains 2 products - 4 factors so it should return 4 partial derivatives
    filter_delta = backprop_filter(inputs_filter,  errors_filter, weights)

    outputs_input_gate = all_layer_outputs["input_gate"]
    errors_input_gate = {"to_filter": filter_delta["input_gate_delta"]}
    input_gate_delta = backprop_input_gate(inputs_input_gate, outputs_input_gate, errors_input_gate, weights)

    outputs_forget_gate = all_layer_outputs["forget_gate"]
    errors_forget_gate = {"to_filter": filter_delta["forget_gate_delta"]}
    forget_gate_delta = backprop_forget_gate(inputs_forget_gate, outputs_forget_gate, errors_forget_gate, weights)

    outputs_unfiltered = all_layer_outputs["output_gate"]
    errors_unfiltered = {"to_filter": lstm_filter["output_gate_delta"],
                          "to_next_filter": next_sample_deltas["filter"]}
    unfiltered_delta = backprop_unfiltered(inputs, outputs_unfiltered, errors_unfiltered, weights)

    return {"dense_deltas": dense_deltas, "recursive_deltas": recursive_deltas}


# gets the error delta it sent to output and the input and returns 2 delta: to pass down and to update its weights
def backprop_dense_layer(inputs, outputs, errors, weights):
    total_delta = sigmoid_derivative(outputs["activation"]) * errors["to_output"]
    return {"total_delta": total_delta, "input_w_delta": np.dot(inputs["from_previous"].T, total_delta)}

def backprop_recursive_layer(inputs, outputs, errors, weights):
    error = np.dot(errors["to_output"], weights["dense"].T) + \
            np.dot(errors["to_next_recursive"], weights["previous_recursive"])
    total_delta = sigmoid_derivative(outputs["activation"]) * error
    return {"total_delta": total_delta,
            "recursive_w_delta": np.dot(inputs["from_recursive"].T, total_delta),
            "input_w_delta": np.dot(inputs["from_previous"].T, total_delta)}

'''

# backprop through time rnn layer ; takes: its raw output, all the errors deltas sent to its successors
# returns: the overall error delta to pass to its predecessors and the deltas to update its own weights
def backprop_recursive_layer(inputs, outputs, errors, weights):
    error = np.dot(errors["to_output"], weights["dense"].T) + np.dot(errors["to_next_recursive"], weights["previous_recursive"])
    total_delta = sigmoid_derivative(outputs["activation"]) * error
    return {"total_delta": total_delta,
            "recursive_w_delta": np.dot(inputs["from_recursive"].T, total_delta),
            "input_w_delta": np.dot(inputs["from_previous"].T, total_delta)}

# back prop one sample unit through all layers :because it's recursive it takes possible deltas from successor
# samples fed forward, just as the feed forward takes recursive outputs from previous samples
def back_prop_sample(inputs, all_layer_outputs, target, next_sample_deltas, weights):
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

# back propagates a sequence of samples - we don't pass delta from previous sequence here
def back_prop_sequence(inputs_seq, outputs_seq, target_seq, weights):
    # dense deltas are not going to be used so no init is needed
    init_recursive_deltas = {"total_delta": zeros((1, recursive_size)),
                             "recursive_w_delta": zeros_like(weights["previous_recursive"]),
                             "input_w_delta": zeros_like(weights["recursive"])}
    init_dense_deltas = {"total_delta": 0, "input_w_delta": zeros_like(weights["dense"])}

    all_deltas_seq = [{"dense_deltas": init_dense_deltas, "recursive_deltas": init_recursive_deltas}]
    for i in range(1, len(inputs_seq) + 1):
        deltas = back_prop_sample(inputs_seq[-i], outputs_seq[-i], target_seq[-i], all_deltas_seq[-1], weights)
        all_deltas_seq.append(deltas.copy())
    weights["log_loss"] += compute_loss_seq(target_seq, [x['from_dense']['activation'][0][0] for x in outputs_seq])
    return all_deltas_seq[1:]

# update weights with a seq  of deltas coresponding to a sequence of inputs
def update_net_weights(all_deltas_seq, weights):
    for all_deltas in all_deltas_seq:
        weights["recursive"] += learning_rate * np.clip(all_deltas["recursive_deltas"]["input_w_delta"], -10, 10)
        weights["recursive_bias"] += learning_rate * np.clip(all_deltas["recursive_deltas"]["total_delta"], -10, 10)
        weights["dense"] += learning_rate * np.clip(all_deltas["dense_deltas"]["input_w_delta"], -10, 10)
        weights["dense_bias"] += learning_rate * np.clip(all_deltas["dense_deltas"]["total_delta"], -10, 10)
        weights["previous_recursive"] += np.clip(learning_rate * all_deltas["recursive_deltas"]["recursive_w_delta"], -10, 10)

def train_net(weights):
    for i in range(n_samples):

        input_1_binary, input_2_binary, target_binary, input_1, input_2 = generate_random_sample()
        input_seq = [{"from_previous": np.array([x]),
                      "from_recursive": zeros((1, recursive_size))} for x in zip(input_1_binary, input_2_binary)]
        update_net_weights(back_prop_sequence(input_seq, feed_forward_sequence(input_seq, weights), target_binary, weights), weights)

        if i % print_every == 0:
            print 100 * '#' + " sample {} ".format(i)
            print "loss is {}".format(weights["log_loss"]/(i*n_bit))
            print " For training sample: {0} + {1} = {2}".format(input_1, input_2, input_1+ input_2 )
            result = list(reversed([x["from_dense"]["activation"] for x in feed_forward_sequence(input_seq, weights)]))
            print " Result is {}".format(sum([pow(2, n_bit - i - 1) * round(result[i]) for i in range(n_bit)]))

train_net(weights)
'''