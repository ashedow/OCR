import random

def init_bias_const(neuron_num):
  return 0.0
  
def init_bias_rand(neuron_num):
  return random.uniform(-1.0, 1.0)

def init_weight_const(neuron_num, input_num):
  return 1.0

def init_weight_rand(neuron_num, input_num):
  return random.uniform(-1.0, 1.0)

def init_weight_identity(neuron_num, input_num):
  return 1.0 if neuron_num == input_num else 0.0


def gen_input_vec(length):
  result = list()
  for i in range(0, length):
    result.append(random.uniform(0.0, 1.0))
  return result


# num_inputs is number of intputs for every neuron
def create_layer(number_of_neurons, number_of_inputs, init_bias_func, init_weight_func):
  layer = dict()
  layer["weights"] = list() # mat
  layer["biases"] = list()  # vec

  for i in range(0, number_of_neurons):
    layer["biases"].append(init_bias_func(i));
    for j in range(0, number_of_inputs):
      layer["weights"].append(init_weight_func(i, j))
  return layer


# num_inputs is number of intputs for every neuron
# TODO: A real network can have different number of neurons in every layer
def create_neural_network(num_inputs, num_hidden_layers, num_neurons_in_hidden_layers, num_output_neurons):
  nn = list()
  for i in range(0, num_hidden_layers):
    nn.append(create_layer(num_neurons_in_hidden_layers, num_inputs, init_bias_rand, init_weight_rand))
  nn.append(create_layer(num_output_neurons, num_inputs, init_bias_rand, init_weight_rand))
  return nn


'''
import dump
layer = create_layer(4, 4, init_bias_rand, init_weight_rand)
inputVec = gen_input_vec(4)
dump.print_layer(layer, inputVec, activate_layer(inputVec, layer, sigmoid))
'''

