from lstm_nice import *


# test set up
n_bit = 3
largest_input_number = pow(2, n_bit) / 2
# RNN params
input_dim = 2
output_dim = 1
recursive_size = 3
learning_rate = .1

# mock data set init everything to 1 for simple minimal test of correct values

weights = { # layer for computing raw lstm values - to be filtered using the input gate
            "input_to_lstm_unfiltered": np.ones((input_dim, recursive_size)),
            "previous_lstm_output_to_lstm_unfiltered": np.ones((recursive_size, recursive_size)),
            "lstm_unfiltered_bias": zeros((1, recursive_size)),

            "input_to_input_gate": np.ones((input_dim, recursive_size))/2,
            "previous_lstm_output_to_input_gate": np.ones((recursive_size, recursive_size))/2,
            "input_gate_bias": zeros((1, recursive_size)),

            "input_to_forget_gate": np.ones((input_dim, recursive_size))/4,
            "previous_lstm_output_to_forget_gate": np.ones((recursive_size, recursive_size))/4,
            "forget_gate_bias": zeros((1, recursive_size)),

            "input_to_output_gate": np.ones((input_dim, recursive_size))/5,
            "previous_lstm_output_to_output_gate": np.ones((recursive_size, recursive_size))/5,
            "output_gate_bias": zeros((1, recursive_size)),

            # output layer weights
            "dense": np.ones((recursive_size, output_dim)),
            "dense_bias": zeros((1,output_dim)),
            # the associated metrics with this set of weights' values
            "log_loss":0 }

def test_feed_forward_network():
    inputs = {"from_input": np.array([[1, 1]]),
              "from_previous_lstm_filter": np.ones((1, recursive_size)),
              "from_previous_lstm": np.ones((1, recursive_size))}
    expected_unfiltered = {"activation": np.array([[tanh(input_dim + recursive_size) for _ in range(recursive_size)]])}
    expected_input_gate = {
        "activation": np.array([[sigmoid((input_dim + recursive_size) / 2.0) for _ in range(recursive_size)]])}
    expected_forget_gate = {
        "activation": np.array([[sigmoid((input_dim + recursive_size) / 4.0) for _ in range(recursive_size)]])}
    expected_lstm_filtered_output = tanh(expected_unfiltered["activation"] * expected_input_gate["activation"] + \
                                         inputs["from_previous_lstm_filter"] * expected_forget_gate["activation"])
    expected_output_gate = {
        "activation": np.array([[sigmoid((input_dim + recursive_size) / 5.0) for _ in range(recursive_size)]])}
    expected_lstm_output = expected_lstm_filtered_output * expected_output_gate["activation"]
    expected_dense = {"activation": np.array([[sigmoid(np.sum(expected_lstm_output))]])}
    actual_output = feed_forward_network(inputs, weights)
    assert list(expected_dense["activation"]) == list(actual_output["dense"]["activation"])


def test_feed_forward_network_seq():

    inputs_sample1 = {"from_input": np.array([[1, 1]]),
              "from_previous_lstm_filter": np.ones((1, recursive_size)),
              "from_previous_lstm": np.ones((1, recursive_size))}
    all_layer_outputs_1 = feed_forward_network(inputs_sample1, weights)
    inputs_sample2 = {"from_input": np.array([[1, 1]]),
              "from_previous_lstm_filter": all_layer_outputs_1["filter"],
              "from_previous_lstm": all_layer_outputs_1["lstm"]}
    all_layer_outputs_2 = feed_forward_network(inputs_sample2, weights)
    all_layer_output_seq = feed_forward_sequence([inputs_sample1, inputs_sample2], weights)
    assert list(all_layer_output_seq[1]["dense"]["activation"][0]) == list(all_layer_outputs_2["dense"]["activation"][0])

def test_backprop_sample():  # inputs, all_layer_outputs, target, next_sample_deltas, weights
    inputs = {"from_input": np.array([[1, 1]]),
              "from_previous_lstm_filter": np.ones((1, recursive_size)),
              "from_previous_lstm": np.ones((1, recursive_size))}
    target = 1
    next_sample_deltas = {"unfiltered": {"total_delta":np.zeros((1,recursive_size))} ,
                          "input_gate": {"total_delta":np.zeros((1,recursive_size))},
                          "forget_gate":  {"total_delta":np.zeros((1,recursive_size))},
                          "output_gate":  {"total_delta":np.zeros((1,recursive_size))},
                          "dense":  {"total_delta":np.zeros((1,recursive_size))},
                          "filter":  {"total_delta":np.zeros((1,recursive_size))}}
    next_sample_outputs = {"forget_gate": {"activation": np.zeros((1,recursive_size)) } }
    all_layer_outputs = feed_forward_network(inputs, weights)
    error_output = all_layer_outputs["dense"]["activation"] - target
    # total delta is always of size: (1, # layer outputs)
    # total deltas are averaged using weights from backwarding layer
    # specific input weights delta is
    total_delta = sigmoid_derivative(all_layer_outputs["dense"]["activation"]) * error_output
    dense_deltas = {"total_delta": total_delta,
                             "input_w_delta": total_delta * all_layer_outputs["lstm"]}
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

    filter_delta = {"total_delta": lstm_deltas["total_delta"] *
                                             all_layer_outputs["output_gate"]["activation"] *
                                             tanh_derivative(all_layer_outputs["filter"] ) +
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
    expected_outputs =  {"dense": dense_deltas,
                        "output_gate": output_gate_delta,
                        "filter": filter_delta,
                        "input_gate": input_gate_delta,
                        "forget_gate": forget_gate_delta,
                        "unfiltered": unfiltered_delta,
                        "lstm": lstm_deltas}

    actual_outputs = back_prop_sample(inputs, all_layer_outputs, next_sample_outputs, target, next_sample_deltas, weights)

    assert len(expected_outputs.keys()) == len(actual_outputs.keys())
    for key in actual_outputs.keys():
        for key2 in expected_outputs[key]:
            assert expected_outputs[key][key2].tolist() == actual_outputs[key][key2].tolist()

def test_backprop_sequence():
    # feed forward 2 samples
    inputs_sample1 = {"from_input": np.array([[1, 1]]),
              "from_previous_lstm_filter": np.ones((1, recursive_size)),
              "from_previous_lstm": np.ones((1, recursive_size))}
    all_layer_outputs_1 = feed_forward_network(inputs_sample1, weights)
    inputs_sample2 = {"from_input": np.array([[1, 1]]),
              "from_previous_lstm_filter": all_layer_outputs_1["filter"],
              "from_previous_lstm": all_layer_outputs_1["lstm"]}
    all_layer_outputs_2 = feed_forward_network(inputs_sample2, weights)
    # back prop sample 2

    target_2 = 1
    next_sample_deltas_2 = {"unfiltered": {"total_delta":np.zeros((1,recursive_size))} ,
                          "input_gate": {"total_delta":np.zeros((1,recursive_size))},
                          "forget_gate":  {"total_delta":np.zeros((1,recursive_size))},
                          "output_gate":  {"total_delta":np.zeros((1,recursive_size))},
                          "dense":  {"total_delta":np.zeros((1,recursive_size))},
                          "filter":  {"total_delta":np.zeros((1,recursive_size))}}
    next_sample_outputs_2 = {"forget_gate": {"activation": np.zeros((1,recursive_size)) } }
    deltas_2 = back_prop_sample(inputs_sample2, all_layer_outputs_2, next_sample_outputs_2, target_2,
                                next_sample_deltas_2, weights)
    # back prop sample 1
    target_1 = 1
    next_sample_deltas_1 = deltas_2
    next_sample_outputs_1= all_layer_outputs_2
    deltas_1 = back_prop_sample(inputs_sample1, all_layer_outputs_1, next_sample_outputs_1, target_1,
                                next_sample_deltas_1, weights)
    # (inputs_seq, outputs_seq, target_seq, weights)
    deltas_seq = back_prop_sequence([inputs_sample1, inputs_sample2], [all_layer_outputs_1, all_layer_outputs_2],
                                    [target_1, target_2],  weights)
    for key in deltas_2.keys():
        for key2 in deltas_2[key]:
            assert deltas_2[key][key2].tolist() == deltas_seq[0][key][key2].tolist()
    for key in deltas_1.keys():
        for key1 in deltas_1[key]:
            assert deltas_1[key][key1].tolist() == deltas_seq[1][key][key1].tolist()



