import dump
import mymath

# deprecated
def activate_layer(input_vec, layer, activation_func):
  weightedSumVec = mymath.multiply_mv(layer["weights"], input_vec)
  biasedSumVec = mymath.vec_sum(weightedSumVec, layer["biases"])
  activationVec = list()
  for i in range(0, len(biasedSumVec)):
    activationVec.append(activation_func(biasedSumVec[i]))
  return activationVec


def sum_layer(input_vec, layer):
  weightedSumVec = mymath.multiply_mv(layer["weights"], input_vec)
  biasedSumVec = mymath.vec_sum(weightedSumVec, layer["biases"])
  return biasedSumVec


def feedforward(input_vec, nn, activation_func, debug=False):
  if len(nn) == 0:
    raise ValueError("Feedforward to empty neural network")
  result = dict()
  result["activations"] = list()
  result["weighted_inputs"] = list()
  activationVec = input_vec
  for layerIndex in range(0, len(nn)):
    nInputs = len(nn[layerIndex]["weights"]) / len(nn[layerIndex]["biases"])
    if nInputs != len(activationVec):
      raise ValueError("NN Layer {} number of inputs doesn't match the length of the data vector.".format(layerIndex))
    if debug:
      inputVec = activationVec
    weightedSum = sum_layer(activationVec, nn[layerIndex])
    result["weighted_inputs"].append(weightedSum)
    activationVec = mymath.vectorized_func(weightedSum, activation_func)
    result["activations"].append(activationVec)
    if debug:
      dump.print_layer(nn[layerIndex], layerIndex, inputVec, activationVec)
  return result

# cost function
# the sum of squared differences between output layer activations and desired outputs over all last layer neurons
def network_error(activation, desired):
  diff = mymath.vec_subtract(desired, activation)
  return mymath.dot(diff, diff)

# which is one of the terms of the derivative (constant factor omitted) of the cost function with respect to some weight incoming to the neuron
def output_neuron_error(activation, desired):
  return activation - desired

def propagate_back(nn, partial_errors, step_num):
  #print(">>> backprop from layer: " + str(-step_num-1))
  if step_num + 1 >= len(nn):
    return False

  layerIndex   = -step_num - 1
  layer        = nn[layerIndex]
  prevLayer    = nn[layerIndex-1]
  layerLen     = len(layer["biases"])
  prevLayerLen = len(prevLayer["biases"])
  result       = [0] * prevLayerLen

  for neuron in range(0, layerLen):
    for prevNeuron in range(0, prevLayerLen):
      #print(">> p {} : n {}".format(prevNeuron, neuron))
      result[prevNeuron] += partial_errors[neuron] * layer["weights"][neuron*prevLayerLen + prevNeuron]
  return result


def learn(nn):
  learningRate = 0.2

  for something in range(0, 1000): # <- loop over training examples
    if something % 2:
      trainingExample = [0.9, 0.8, 0.1, 0.1]
      desired = [0.99, 0.01]
    else:
      trainingExample = [0.1, 0.1, 0.8, 0.9]
      desired = [0.01, 0.99]

    result = feedforward(trainingExample, nn, mymath.sigmoid, True)
    partial_errors = mymath.vectorized_func2(result["activations"][-1], desired, output_neuron_error)

    for i in range(0, len(nn)): # <- for every layer do backprop + gradient descent
      if partial_errors == False:
        raise ValueError("Partial errors unknown, have we tried to propagate back too much?")

      layerIndex = -i - 1 # layerindex is negative, because it's from the end
      correctionTerm = list(partial_errors)
      #print(">>> layer index: ", str(layerIndex))
      #print(">>> partialerrors len: " + str(len(partial_errors)))
      #print(">>> weighted inputs len: " + str(result["weighted_inputs"][layerIndex]))
      correctionTerm = mymath.hadamard(correctionTerm, mymath.vectorized_func(result["weighted_inputs"][layerIndex], mymath.sigmoid_derivative))
      mymath.multiply_vs_in_place(correctionTerm, -learningRate)

      if i + 1 == len(nn):
        prevLayerActivation = list(trainingExample)
      else:
        prevLayerActivation = list(result["activations"][layerIndex-1])

      correctionsMatrixForLayer = list()

      for neuronIndex in range(0, len(nn[layerIndex]["biases"])):
        for prevLayerNeuronIndex in range(0, len(prevLayerActivation)):
          correctionsMatrixForLayer.append(correctionTerm[neuronIndex] * prevLayerActivation[prevLayerNeuronIndex])

      # applying corrections:
      for dendrite in range(0, len(nn[layerIndex]["weights"])):
        nn[layerIndex]["weights"][dendrite] += correctionsMatrixForLayer[dendrite]

      partial_errors = propagate_back(nn, partial_errors, i) # <- backprop, prepare for the next loop iteration
