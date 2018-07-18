import random

def init_bias_const(neuron_num):
  return 0.0
  
def init_bias_rand(neuron_num):
  return random.uniform(-1.0, 1.0)

def init_weight_const(neuron_num, input_num):
  return 1.0

def init_weight_rand(neuron_num, input_num):
  return random.uniform(-0.2, 0.2)

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

# neurons is a list of natural numbers representing number of neurons at every layer
# neurons[0] is the number of inputs to neural network (i.e. length of the data example)
def create_neural_network(neurons):
  nn = list()
  for i in range(1, len(neurons)):
    nn.append(create_layer(neurons[i], neurons[i-1], init_bias_const, init_weight_rand))
  return nn

'''
import dump
layer = create_layer(4, 4, init_bias_rand, init_weight_rand)
inputVec = gen_input_vec(4)
dump.print_layer(layer, inputVec, activate_layer(inputVec, layer, sigmoid))
'''

