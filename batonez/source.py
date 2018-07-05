def dot(a, b):
    length = len(a)
    if length != len(b):
        raise ValueError("Cannot dot product vector(" + str(length) + ") with vector(" + str(len(b)) + ")")

    result = 0
    for i in range(0, length):
        result += a[i] * b[i]
    return result

'''
one = [1,2,-3]
two = [1,2,3]
print dot(one,two)
'''
