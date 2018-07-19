from __future__ import print_function
import gen
import neural 
import mymath
import dump

nn = gen.create_neural_network([4, 6, 2])
result = neural.feedforward(gen.gen_input_vec(4), nn, mymath.sigmoid, True)
print("Activations:")
dump.print_vectors(result["activations"])
print("\nWeighted inputs:")
dump.print_vectors(result["weighted_inputs"])

