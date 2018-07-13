from __future__ import print_function
import gen
import neural 

nn = gen.create_neural_network(4, 3, 4, 2)
result = neural.feedforward(gen.gen_input_vec(4), nn, True)

