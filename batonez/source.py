def dot(a, b, length = 0, afrom = 0, bfrom = 0):
  if length <= 0:
    length = min(len(a), len(b))
  if (bfrom + length > len(b)) or (afrom + length > len(a)):
    raise ValueError("Cannot dot product: operand is too short")

  result = 0
  for i in range(0, length):
    result += a[afrom + i] * b[bfrom + i]
  return result


def multiplyMV(mat, vec):
  veclen = len(vec)
  matlen = len(mat)
  if matlen % veclen:
    raise ValueError("Cannot multiply matrix by vector: wrong dimensions")

  rows = matlen / veclen
  result = list()
  for row in range(0, rows):
    result.append( dot(mat, vec, veclen, row*veclen, 0))

  return result

'''
one = [444, 11, 5, 1,2,-3, 123, 123, 44]
two = [145, 1,2,3, 124, 11, 445]
print dot(one,two, 3, 3, 1)
'''

'''
mat = [1,2,-3, 2,2,2, 3,3,-1]
vec = [1,2,3]
print multiplyMV(mat, vec)
'''
