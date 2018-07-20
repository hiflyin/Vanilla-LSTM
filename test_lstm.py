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

def test_feed_forward_to_lstm_unfiltered():
    inputs = {"from_input": np.array([[1,1]]),
              "from_previous_lstm_filter": np.ones((1, recursive_size)),
              "from_previous_lstm": np.ones((1, recursive_size))}
    # provided all value are set to 1 - except biases which are set to 0, we should get
    expected_output =  {"activation": np.array([[tanh(input_dim + recursive_size) for _ in range(recursive_size)]])}
    actual_output = feed_forward_to_lstm_unfiltered(inputs, weights)

    assert list(expected_output["activation"][0]) == list(actual_output['activation'][0])

def test_feed_forward_to_input_gate():
    inputs = {"from_input": np.array([[1,1]]),
              "from_previous_lstm_filter": np.ones((1, recursive_size)),
              "from_previous_lstm": np.ones((1, recursive_size))}
    # provided all value are set to 1 - except biases which are set to 0, we should get
    expected_output = {"activation": np.array([[sigmoid((input_dim + recursive_size)/2.0) for _ in range(recursive_size)]])}
    actual_output = feed_forward_to_input_gate(inputs, weights)

    assert list(expected_output["activation"][0]) == list(actual_output['activation'][0])

def test_feed_forward_to_forget_gate():
    inputs = {"from_input": np.array([[1,1]]),
              "from_previous_lstm_filter": np.ones((1, recursive_size)),
              "from_previous_lstm": np.ones((1, recursive_size))}
    # provided all value are set to 1 - except biases which are set to 0, we should get
    expected_output = {"activation": np.array([[sigmoid((input_dim + recursive_size)/4.0) for _ in range(recursive_size)]])}
    actual_output = feed_forward_to_forget_gate(inputs, weights)

    assert list(expected_output["activation"][0]) == list(actual_output['activation'][0])

    #return inputs["from_lstm_unfiltered"]["activation"]*inputs["from_input_gate"]["activation"] + \
     #      inputs["from_previous_lstm_filter"]["activation"] * inputs["from_forget_gate"]["activation"]
def test_feed_forward_through_lstm_filter():
    inputs = {"from_input": np.array([[1,1]]),
              "from_previous_lstm_filter": np.ones((1, recursive_size)),
              "from_previous_lstm": np.ones((1, recursive_size))}
    expected_unfiltered = {"activation": np.array([[tanh(input_dim + recursive_size) for _ in range(recursive_size)]])}
    expected_input_gate = {"activation": np.array([[sigmoid((input_dim + recursive_size)/2.0) for _ in range(recursive_size)]])}
    expected_forget_gate = {"activation": np.array([[sigmoid((input_dim + recursive_size)/4.0) for _ in range(recursive_size)]])}

    expected_lstm_filtered_output = tanh(expected_unfiltered["activation"]*expected_input_gate["activation"] + \
                                    inputs["from_previous_lstm_filter"]*expected_forget_gate["activation"])
    actual_output = feed_forward_through_lstm_filter(inputs, weights)
    assert list(expected_lstm_filtered_output[0]) == list(actual_output[0])

def test_feed_forward_to_output_gate():
    inputs = {"from_input": np.array([[1,1]]), "from_previous_lstm": np.ones((1, recursive_size))}
    # provided all value are set to 1 - except biases which are set to 0, we should get
    expected_output = [sigmoid((input_dim + recursive_size)/5.0) for _ in range(recursive_size)]
    actual_output = list(feed_forward_to_output_gate(inputs, weights)['activation'][0])

    assert expected_output==actual_output

def test_feed_forward_through_output_gate():
    inputs = {"from_input": np.array([[1,1]]),
              "from_previous_lstm_filter": np.ones((1, recursive_size)),
              "from_previous_lstm": np.ones((1, recursive_size))}
    expected_unfiltered = {"activation": np.array([[tanh(input_dim + recursive_size) for _ in range(recursive_size)]])}
    expected_input_gate = {"activation": np.array([[sigmoid((input_dim + recursive_size)/2.0) for _ in range(recursive_size)]])}
    expected_forget_gate = {"activation": np.array([[sigmoid((input_dim + recursive_size)/4.0) for _ in range(recursive_size)]])}
    expected_lstm_filtered_output = tanh(expected_unfiltered["activation"]*expected_input_gate["activation"] + \
                                    inputs["from_previous_lstm_filter"]*expected_forget_gate["activation"])
    expected_output_gate = {"activation": np.array([[sigmoid((input_dim + recursive_size) / 5.0) for _ in range(recursive_size)]])}
    expected_output = expected_lstm_filtered_output*expected_output_gate["activation"]
    assert list(expected_output[0]) == list(feed_forward_through_output_gate(inputs, weights)[0])


def _test_feed_forward_dense_layer():

    input_values = [sigmoid(input_dim + recursive_size) for _ in range(recursive_size)]
    # provided all value are set to 1 - except biases which are set to 0, we should get
    expected_output = [sigmoid(sum(input_values)) ]
    actual_output = list(feed_forward_dense_layer({"from_previous": np.array([input_values])}, weights)['activation'][0])
    assert expected_output==actual_output

def _test_backprop_dense_layer():

    input_values = [sigmoid(input_dim + recursive_size) for _ in range(recursive_size)]
    inputs = {"from_previous": np.array([input_values])}
    expected_outputs = {"activation": np.array([[sigmoid(sum(input_values))]]) }
    fake_error = {"to_output": 1}
    expected_total_delta = sigmoid_derivative(expected_outputs["activation"])*fake_error["to_output"]
    expected_input_w_delta = expected_total_delta*np.array([input_values])
    actual_outputs = backprop_dense_layer(inputs, expected_outputs, fake_error, weights)
    assert list(actual_outputs["total_delta"][0]) == list(expected_total_delta[0])
    assert list(actual_outputs["input_w_delta"].T[0]) == list(expected_input_w_delta[0])

def _test_backprop_recursive_layer():

    inputs = {"from_previous": np.array([[1,1]]), "from_recursive": np.ones((1, recursive_size))}
    # provided all value are set to 1 - except biases which are set to 0, we should get
    output_values = [sigmoid(input_dim + recursive_size) for _ in range(recursive_size)]
    expected_outputs = {"activation": np.array([output_values]) }
    expected_dense_delta = sigmoid_derivative( np.array([[sigmoid(sum(output_values))]]))

    expected_deltas_passed = {"to_output": expected_dense_delta, "to_next_recursive": zeros((1, recursive_size))}
    expected_total_delta = sigmoid_derivative(expected_outputs["activation"]) * expected_dense_delta
    actual_outputs = backprop_recursive_layer(inputs, expected_outputs,  expected_deltas_passed, weights)

    assert list(actual_outputs["total_delta"][0]) == list(expected_total_delta[0])




