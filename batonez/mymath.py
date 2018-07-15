import math

def vec_sum(a, b):
  if len(a) != len(b):
    raise ValueError("Cannot sum two vectors of the different length")
  result = list()
  for i in range(0, len(a)):
    result.append(a[i] + b[i])
  return result


def dot(a, b, length = 0, aFrom = 0, bFrom = 0):
  if length <= 0:
    length = min(len(a), len(b))
  if (bFrom + length > len(b)) or (aFrom + length > len(a)):
    raise ValueError("Cannot dot product: operand is too short")

  result = 0
  for i in range(0, length):
    result += a[aFrom + i] * b[bFrom + i]
  return result


def multiply_mv(mat, vec):
  vecLen = len(vec)
  matLen = len(mat)
  if matLen % vecLen:
    raise ValueError("Cannot multiply matrix by vector: wrong dimensions")

  rows = matLen / vecLen
  result = list()
  for row in range(0, rows):
    result.append(dot(mat, vec, vecLen, row*vecLen, 0))

  return result


def hadamard(a, b):
  if (len(a) != len(b)):
    raise ValueError("Cannot hadamard product two vectors of the different length")
  result = list()
  for i in range(0, len(a)):
    result.append(a[i] * b[i])
  return result


def sigmoid(x):
  return 1.0/(1.0 + pow(math.e, -x))


'''
print hadamard([1, 2, 3], [3, 2, 1])
'''

'''
print str(sigmoid(0))
'''

'''
one = [444, 11, 5, 1,2,-3, 123, 123, 44]
two = [145, 1,2,3, 124, 11, 445]
print dot(one,two, 3, 3, 1)
'''

'''
mat = [1,2,-3, 2,2,2, 3,3,-1]
vec = [1,2,3]
print multiply_mv(mat, vec)
'''
