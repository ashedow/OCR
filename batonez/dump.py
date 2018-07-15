from __future__ import print_function

def print_layer(layer, layer_index=None, input_vec=None, activation_vec=None):
  nNeurons = len(layer["biases"])
  nInputs = len(layer["weights"]) / len(layer["biases"])
  print("\n===== Layer {} dump ============================================".format(layer_index))
  print("Number of neurons: " + str(nNeurons))
  print("Number of inputs for every neuron: " + str(nInputs))

  for j in range(0, nInputs):
    print("--------", end="")
  if input_vec:
    print("-----------", end="")
  print("---------", end="")
  if activation_vec:
    print("-----------", end="")
  print("")

  print("in         " if input_vec else "  ", end="")
  print("weights                            ", end="")
  print("biases     ", end="")
  print("a")
  
  for j in range(0, nInputs):
    print("--------", end="")
  if input_vec:
    print("-----------", end="")
  print("---------", end="")
  if activation_vec:
    print("-----------", end="")
  print("")

  inputVecLen = len(input_vec) if input_vec else 0
  numRows = inputVecLen if  (inputVecLen > nNeurons) else nNeurons

  for i in range(0, numRows):
    if i < inputVecLen:
      print("{: >+3.3f}  *  ".format(input_vec[i]), end="")
    else:
      print("        *  ", end="")
    if i < nNeurons:
      for j in range(0, nInputs):
        print("{: >+3.3f}".format(layer["weights"][j + i * nInputs]) + "  ", end="")
      print("|  {: >+3.3f}".format(layer["biases"][i]), end=("" if activation_vec else "\n"))
      if activation_vec:
        print("  =  {: >+3.3f}".format(activation_vec[i]))
    else:
      print("")
  print("===== End of layer {} dump =====================================".format(layer_index))


def print_neural_network(nn):
  for i in range(0, len(nn)):
    print_layer(nn[i], i) 

