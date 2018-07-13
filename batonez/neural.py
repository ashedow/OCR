import dump
import mymath

def activate_layer(input_vec, layer, activation_func):
  weightedSumVec = mymath.multiply_mv(layer["weights"], input_vec)
  biasedSumVec = mymath.vec_sum(weightedSumVec, layer["biases"])
  activationVec = list()
  for i in range(0, len(biasedSumVec)):
    activationVec.append(activation_func(biasedSumVec[i]))
  return activationVec

def feedforward(input_vec, nn, debug=False):
  if len(nn) == 0:
    raise ValueError("Feedforward to empty neural network")
  dataVec = input_vec
  for layerIndex in range(0, len(nn)):
    nInputs = len(nn[layerIndex]["weights"]) / len(nn[layerIndex]["biases"])
    if nInputs != len(dataVec):
      raise ValueError("NN Layer {} number of inputs doesn't match the length of the data vector.".format(layerIndex))
    if debug:
      inputVec = dataVec
    dataVec = activate_layer(dataVec, nn[layerIndex], mymath.sigmoid)
    if debug:
      dump.print_layer(nn[layerIndex], layerIndex, inputVec, dataVec)
  return dataVec 

