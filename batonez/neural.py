import dump
import mymath

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

