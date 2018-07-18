import math

def vec_sum(a, b):
  if len(a) != len(b):
    raise ValueError("Cannot sum two vectors of the different length")
  result = list()
  for i in range(0, len(a)):
    result.append(a[i] + b[i])
  return result


def vec_subtract(a, b):
  if len(a) != len(b):
    raise ValueError("Cannot sum two vectors of the different length")
  result = list()
  for i in range(0, len(a)):
    result.append(a[i] - b[i])
  return result


def transpose_in_place(mat, num_rows, num_cols):
  if len(mat) / num_rows != num_cols:
    raise ValueError("transpose: matrix dimensions do not match the input array length")
  for i in range(0, num_rows):
    for j in range(0, i):
      temp = mat[num_cols*i + j]
      mat[num_cols*i + j] = mat[num_cols*j + i]
      mat[num_cols*j + i] = temp
  return mat


def transpose(mat, num_rows, num_cols):
  result = list(mat)
  return transpose_in_place(result, num_rows, num_cols)


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


def vectorized_func(vec_in, func):
  result = list()
  for i in range(0, len(vec_in)):
    result.append(func(vec_in[i]))
  return result

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

'''
import dump
mat = [1,2,3,4,5,6,7,8,9]
dump.print_mat(transpose(mat, 3, 3), 3)
print("----")
dump.print_mat(mat, 3)
'''
