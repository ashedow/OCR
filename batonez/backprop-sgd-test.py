from __future__ import print_function
import mymath
import gen
import neural 
import dump

nn = gen.create_neural_network([4, 10, 2])
neural.learn(nn)

print("\n====\nAFTER LEARNED 1:=====\n")
example = [0.9, 0.8, 0.1, 0.1]
result = neural.feedforward(example, nn, mymath.sigmoid, True)

print("\n====\nAFTER LEARNED 2:=====\n")
example = [0.1, 0.1, 0.8, 0.9]
result = neural.feedforward(example, nn, mymath.sigmoid, True)
