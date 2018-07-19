
# to do:

# 4 nice dict print
# 5 gradient check

from rnn_corr import *
# test functions - not asserting correct results - just making sure they run with correct dimensions
# set test constants


n_bit = 3
largest_input_number = pow(2, n_bit) / 2
recursive_size = 3
sample_data= np.array([[0,1]])

error = 1
# init test weights to 1 for simple test of correct values
weights = {"recursive": np.ones((input_dim, recursive_size)),
           "previous_recursive": np.ones((recursive_size, recursive_size)),
           "recursive_bias": zeros((1, recursive_size)),
           "dense": np.ones((recursive_size, output_dim)),
           "dense_bias": zeros((1,output_dim)),
           "log_loss":0
          }


print generate_random_sample()
print sigmoid(np.array([range(recursive_size)]))
print sigmoid_derivative(np.array([range(recursive_size)]))

inputs_recursive = {"from_previous": sample_data, "from_recursive": np.ones((1,recursive_size))}
print feed_forward_recursive_layer(inputs_recursive, weights)

inputs_dense = {"from_previous": ones((1,recursive_size))}
print feed_forward_dense_layer(inputs_dense, weights)

outputs_dense = {"raw": 0, "activation": np.array([[0]])}
errors_dense = {"to_output": 1}
print backprop_dense_layer(inputs_dense, outputs_dense, errors_dense, weights)

outputs_recursive = {"raw": ones((1,recursive_size))/2, "activation": ones((1,recursive_size))/2}
# assume there was no error sent to next hidden layer
errors_recursive_case1 = {"to_output": 1, "to_next_recursive": zeros((1,recursive_size))}
# assume there was no error sent to next layer (the output dense layer)
errors_recursive_case2 = {"to_output": 0, "to_next_recursive": ones((1,recursive_size))}
print backprop_recursive_layer(inputs_recursive,  outputs_recursive, errors_recursive_case1, weights)
print backprop_recursive_layer(inputs_recursive,  outputs_recursive, errors_recursive_case2, weights)

#print feed_forward_network(inputs_recursive)
all_layer_outputs = {"from_dense":outputs_dense, "from_recursive":outputs_recursive}
correct_output = 1
next_sample_recursive_deltas = {"total_delta": zeros((1,recursive_size)), "recursive_w_delta" : None, "input_w_delta" : None}
next_sample_deltas = {"dense_deltas":None, "recursive_deltas":next_sample_recursive_deltas}
print back_prop_network(inputs_recursive, all_layer_outputs, correct_output, next_sample_deltas, weights)

inputs_seq = [inputs_recursive, inputs_recursive, inputs_recursive]
outputs_seq = [all_layer_outputs, all_layer_outputs, all_layer_outputs]
target_seq = [correct_output, correct_output, correct_output ]
print feed_forward_network_sequence(inputs_seq, weights)
print back_prop_network_sequence(inputs_seq, outputs_seq, target_seq, weights)


update_network_weights(back_prop_network_sequence(inputs_seq, outputs_seq, target_seq, weights), weights)
