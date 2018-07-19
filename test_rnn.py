from rnn_nice import *

# test set up
n_bit = 3
largest_input_number = pow(2, n_bit) / 2
# RNN params
input_dim = 2
output_dim = 1
recursive_size = 3
learning_rate = .1
# mock data set init everything to 1 for simple minimal test of correct values
sample_input1 = np.array([[1,1]])
inputs_recursive = {"from_previous": sample_input1, "from_recursive": np.ones((1,recursive_size))}
inputs_dense = {"from_previous": ones((1,recursive_size))}
errors_dense1 = {"to_output": 1}
weights = {"recursive": np.ones((input_dim, recursive_size)),
           "previous_recursive": np.ones((recursive_size, recursive_size)),
           "recursive_bias": zeros((1, recursive_size)),
           "dense": np.ones((recursive_size, output_dim)),
           "dense_bias": zeros((1,output_dim)),
           "log_loss":0
          }

def test_feed_forward_recursive_layer():

    inputs = {"from_previous": np.array([[1,1]]), "from_recursive": np.ones((1, recursive_size))}
    # provided all value are set to 1 - except biases which are set to 0, we should get
    expected_output = [sigmoid(input_dim + recursive_size) for _ in range(recursive_size)]
    actual_output = list(feed_forward_recursive_layer(inputs, weights)['activation'][0])

    assert expected_output==actual_output

def test_feed_forward_dense_layer():

    input_values = [sigmoid(input_dim + recursive_size) for _ in range(recursive_size)]
    # provided all value are set to 1 - except biases which are set to 0, we should get
    expected_output = [sigmoid(sum(input_values)) ]
    actual_output = list(feed_forward_dense_layer({"from_previous": np.array([input_values])}, weights)['activation'][0])
    assert expected_output==actual_output

def test_backprop_dense_layer():

    input_values = [sigmoid(input_dim + recursive_size) for _ in range(recursive_size)]
    inputs = {"from_previous": np.array([input_values])}
    expected_outputs = {"activation": np.array([[sigmoid(sum(input_values))]]) }
    fake_error = {"to_output": 1}
    expected_total_delta = sigmoid_derivative(expected_outputs["activation"])*fake_error["to_output"]
    expected_input_w_delta = expected_total_delta*np.array([input_values])
    actual_outputs = backprop_dense_layer(inputs, expected_outputs, fake_error, weights)
    assert list(actual_outputs["total_delta"][0]) == list(expected_total_delta[0])
    assert list(actual_outputs["input_w_delta"].T[0]) == list(expected_input_w_delta[0])

def test_backprop_recursive_layer():

    inputs = {"from_previous": np.array([[1,1]]), "from_recursive": np.ones((1, recursive_size))}
    # provided all value are set to 1 - except biases which are set to 0, we should get
    output_values = [sigmoid(input_dim + recursive_size) for _ in range(recursive_size)]
    expected_outputs = {"activation": np.array([output_values]) }
    expected_dense_delta = sigmoid_derivative( np.array([[sigmoid(sum(output_values))]]))

    expected_deltas_passed = {"to_output": expected_dense_delta, "to_next_recursive": zeros((1, recursive_size))}
    expected_total_delta = sigmoid_derivative(expected_outputs["activation"]) * expected_dense_delta
    actual_outputs = backprop_recursive_layer(inputs, expected_outputs,  expected_deltas_passed, weights)

    assert list(actual_outputs["total_delta"][0]) == list(expected_total_delta[0])




