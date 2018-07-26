import numpy as np
from numpy import ones, zeros, zeros_like, log, clip
######################################### THE DATA #########################################
np.random.seed(0)
n_samples, print_every, n_bit = 15000, 1000, 8
largest_input_number = pow(2,n_bit-1) - 1

def generate_random_sample():
    input_1, input_2 = np.random.randint(0, largest_input_number), np.random.randint(0, largest_input_number)
    input_1_binary = [int(x) for x in np.binary_repr(input_1, n_bit)]
    input_2_binary = [int(x) for x in np.binary_repr(input_2, n_bit)]
    target_binary = [int(x) for x in np.binary_repr(input_1 + input_2, n_bit)]
    return list(reversed(input_1_binary)), list(reversed(input_2_binary)), list(reversed(target_binary)), input_1, input_2

############################################# THE RNN #############################################
input_dim, output_dim, recursive_size, learning_rate = 2, 1, 16, -.08
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

            "input_to_output_gate": np.random.standard_normal(size=(input_dim, recursive_size)),
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
def tanh_derivative(x): return 1-tanh(x)*tanh(x)

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
    lstm_output = feed_forward_through_output_gate(inputs, weights)
    return {"activation": sigmoid(np.dot(lstm_output, weights["dense"]) + weights["dense_bias"])}

# feed forward one sample unit through all layers
def feed_forward_network(inputs, weights):

    unfiltered_values = feed_forward_to_lstm_unfiltered(inputs, weights)
    input_gate = feed_forward_to_input_gate(inputs, weights)
    forget_gate = feed_forward_to_forget_gate(inputs, weights)
    filtered_values = feed_forward_through_lstm_filter(inputs, weights)
    output_gate = feed_forward_to_output_gate(inputs, weights)
    lstm_output = feed_forward_through_output_gate(inputs, weights)
    dense = feed_forward_dense_layer(inputs, weights)
    return {"input": inputs,
            "unfiltered": unfiltered_values,
            "input_gate": input_gate,
            "forget_gate" : forget_gate,
            "filter" : filtered_values,
            "output_gate" : output_gate,
            "lstm" : lstm_output,
            "dense": dense}

# feeds forward a sequence of samples..
def feed_forward_sequence(inputs_seq, weights):

    all_samples_output_seq = [{"filter": inputs_seq[0]["from_previous_lstm_filter"],
                               "lstm": inputs_seq[0]["from_previous_lstm"]}]
    for input_unit in inputs_seq:
        input_unit["from_previous_lstm"] = all_samples_output_seq[-1]["lstm"]
        input_unit["from_previous_lstm_filter"] = all_samples_output_seq[-1]["filter"]
        all_samples_output_seq.append(feed_forward_network(input_unit, weights))
    return all_samples_output_seq[1:]

# back propagates a sequence of samples - we don't pass delta from previous sequence here
# recurrences:
# lstm final output sends errors to next unfiltered values and input, forget and output gates - then deltas should go back
# lstm filtered values send errors to next lstm filter - so these deltas should be also passed back
def back_prop_sequence(inputs_seq, outputs_seq, target_seq, weights):

    deltas_seq = [{"unfiltered": {"total_delta": np.zeros((1, recursive_size))},
                            "input_gate": {"total_delta": np.zeros((1, recursive_size))},
                            "forget_gate": {"total_delta": np.zeros((1, recursive_size))},
                            "output_gate": {"total_delta": np.zeros((1, recursive_size))},
                            "dense": {"total_delta": np.zeros((1, recursive_size))},
                            "filter": {"total_delta": np.zeros((1, recursive_size))}}]
    outputs_seq.append({"forget_gate": {"activation": np.zeros((1,recursive_size)) } })
    for i in range(1, len(inputs_seq) + 1):
        deltas_seq.append(back_prop_sample(inputs_seq[-i], outputs_seq[-i-1], outputs_seq[-i], target_seq[-i],
                                           deltas_seq[-1], weights))
    outputs_seq.pop()
    weights["log_loss"] += compute_loss_seq(target_seq, [x['dense']['activation'][0][0] for x in outputs_seq])
    return deltas_seq[1:]

# back prop one sample unit through all layers :because it's recursive it takes possible deltas from successor
# samples fed forward, just as the feed forward takes recursive outputs from previous samples
def back_prop_sample(inputs, all_layer_outputs, next_sample_outputs, target, next_sample_deltas, weights):
    error_output = all_layer_outputs["dense"]["activation"] - target
    total_delta = sigmoid_derivative(all_layer_outputs["dense"]["activation"]) * error_output
    dense_deltas = {"total_delta": total_delta,
                    "input_w_delta": np.dot( all_layer_outputs["lstm"].T, total_delta)}
    lstm_deltas = {"total_delta": dense_deltas["total_delta"] +
                                  np.dot(next_sample_deltas["input_gate"]["total_delta"], weights[
                                      "previous_lstm_output_to_input_gate"].T) +
                                  np.dot(next_sample_deltas["forget_gate"]["total_delta"], weights[
                                      "previous_lstm_output_to_forget_gate"].T) +
                                  np.dot(next_sample_deltas["output_gate"]["total_delta"], weights[
                                      "previous_lstm_output_to_output_gate"].T) +
                                  np.dot(next_sample_deltas["unfiltered"]["total_delta"], weights[
                                      "previous_lstm_output_to_lstm_unfiltered"].T)}

    total_delta = lstm_deltas["total_delta"] * all_layer_outputs["filter"] * \
                  sigmoid_derivative(all_layer_outputs["output_gate"]["activation"])
    output_gate_delta = {"total_delta": total_delta,
                          "input_w_delta": np.dot(inputs["from_input"].T, total_delta),
                          "previous_lstm_w_delta": np.dot(inputs["from_previous_lstm"].T, total_delta)}

    filter_delta = {"total_delta": lstm_deltas["total_delta"] * all_layer_outputs["output_gate"]["activation"] *
                                    tanh_derivative(all_layer_outputs["filter"]) +
                                    next_sample_deltas["filter"]["total_delta"] *
                                    next_sample_outputs["forget_gate"]["activation"]}

    total_delta = filter_delta["total_delta"] * all_layer_outputs["unfiltered"]["activation"] * \
                  sigmoid_derivative(all_layer_outputs["input_gate"]["activation"])
    input_gate_delta = {"total_delta": total_delta,
                         "input_w_delta": np.dot(inputs["from_input"].T, total_delta),
                         "previous_lstm_w_delta": np.dot(inputs["from_previous_lstm"].T, total_delta)}

    total_delta = filter_delta["total_delta"] * next_sample_deltas["forget_gate"]["total_delta"] * \
                  sigmoid_derivative(all_layer_outputs["forget_gate"]["activation"])
    forget_gate_delta = {"total_delta": total_delta,
                          "input_w_delta": np.dot(inputs["from_input"].T, total_delta),
                          "previous_lstm_w_delta": np.dot(inputs["from_previous_lstm"].T, total_delta)}
    total_delta = filter_delta["total_delta"] * all_layer_outputs["input_gate"]["activation"] * \
                  sigmoid_derivative(all_layer_outputs["unfiltered"]["activation"])
    unfiltered_delta = {"total_delta": total_delta,
                         "input_w_delta": np.dot(inputs["from_input"].T, total_delta),
                         "previous_lstm_w_delta": np.dot(inputs["from_previous_lstm"].T, total_delta)}

    return {"dense": dense_deltas,
            "output_gate": output_gate_delta,
            "filter": filter_delta,
            "input_gate": input_gate_delta,
            "forget_gate": forget_gate_delta,
            "unfiltered": unfiltered_delta,
            "lstm": lstm_deltas}

# update weights with a seq  of deltas coresponding to a sequence of inputs
def update_net_weights(all_deltas_seq, weights):
    for all_deltas in all_deltas_seq:
        #print all_deltas["dense"]
        weights["input_to_lstm_unfiltered"] += learning_rate * np.clip(all_deltas["unfiltered"]["input_w_delta"], -10, 10)
        weights["previous_lstm_output_to_lstm_unfiltered"] += \
            learning_rate * np.clip(all_deltas["unfiltered"]["previous_lstm_w_delta"], -10, 10)
        weights["lstm_unfiltered_bias"] += learning_rate * np.clip(all_deltas["unfiltered"]["total_delta"], -10, 10)

        weights["input_to_input_gate"] += learning_rate * np.clip(all_deltas["input_gate"]["input_w_delta"], -10, 10)
        weights["previous_lstm_output_to_input_gate"] += \
            learning_rate * np.clip(all_deltas["input_gate"]["previous_lstm_w_delta"], -10, 10)
        weights["input_gate_bias"] += learning_rate * np.clip(all_deltas["input_gate"]["total_delta"], -10, 10)

        weights["input_to_forget_gate"] += learning_rate * np.clip(all_deltas["forget_gate"]["input_w_delta"], -10, 10)
        weights["previous_lstm_output_to_forget_gate"] += \
            learning_rate * np.clip(all_deltas["forget_gate"]["previous_lstm_w_delta"], -10, 10)
        weights["forget_gate_bias"] += learning_rate * np.clip(all_deltas["forget_gate"]["total_delta"], -10, 10)

        weights["input_to_output_gate"] += learning_rate * np.clip(all_deltas["output_gate"]["input_w_delta"], -10, 10)
        weights["previous_lstm_output_to_output_gate"] += \
            learning_rate * np.clip(all_deltas["output_gate"]["previous_lstm_w_delta"], -10, 10)
        weights["output_gate_bias"] += learning_rate * np.clip(all_deltas["output_gate"]["total_delta"], -10, 10)

        weights["dense"] += learning_rate * np.clip(all_deltas["dense"]["input_w_delta"], -10, 10)
        weights["dense_bias"] += learning_rate * np.clip(all_deltas["dense"]["total_delta"], -10, 10)

def train_net(weights):
    for i in range(n_samples):

        input_1_binary, input_2_binary, target_binary, input_1, input_2 = generate_random_sample()
        input_seq = [{"from_input": np.array([x]),
                      "from_previous_lstm_filter": zeros((1, recursive_size)),
                      "from_previous_lstm": zeros((1, recursive_size))} for x in zip(input_1_binary, input_2_binary)]

        update_net_weights(back_prop_sequence(input_seq, feed_forward_sequence(input_seq, weights), target_binary, weights), weights)

        if i % print_every == 0 and i > 0 :
            #print weights["input_to_lstm_unfiltered"]
            print 100 * '#' + " sample {} ".format(i)
            print "loss is {}".format(weights["log_loss"]/(i*n_bit))
            print " For training sample: {0} + {1} = {2}".format(input_1, input_2, input_1+ input_2 )
            result = list(reversed([x["dense"]["activation"] for x in feed_forward_sequence(input_seq, weights)]))
            print " Result is {}".format(sum([pow(2, n_bit - i - 1) * round(result[i]) for i in range(n_bit)]))

train_net(weights)
