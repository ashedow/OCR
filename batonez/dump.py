from __future__ import print_function

def print_mat(mat, num_cols):
  for i in range(0, len(mat)):
    print("{: >+3.3f}  ".format(mat[i]), end="")
    if i > 0 and (i+1) % num_cols == 0:
      print("")


def print_vectors(vectors):
  j = 0
  done = False
  while not done:
    done = True
    for i in range(0, len(vectors)):
      if j < len(vectors[i]):
        print("{: >+3.3f}   ".format(vectors[i][j]), end="")
        done = False
    print("")
    j = j + 1


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

'''
mat = [1,2,3,4,5,6,7,8,9]
print_mat(mat, 3)
'''
