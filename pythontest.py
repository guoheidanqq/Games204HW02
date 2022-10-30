import numpy as np

#
#  a = np.arange(4).reshape(4,1) # (4,1)
#  b = np.ones(5) # (5,)
# (a + b).shape

# #eisum test
# A = np.array([0, 1, 2])
# A_add = A[:,np.newaxis]
# B = np.array([[ 0,  1,  2,  3],
#               [ 4,  5,  6,  7],
#               [ 8,  9, 10, 11]])
#
# C = A_add * B
# D = C.sum(axis = 1)


# tensor product
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[4, 5, 6], [7, 8, 9]])
z3 = np.tensordot(x, y, axes=0)
print(z3)
print(z3.shape)
