import numpy as np

# ====== Test 1 ======

# A = np.ones((3, 4))

# B = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])

# print(B)
# print(B[:,2].reshape(-1, 1))
# print(B[1,:])

# ====== Test 2 ======

# from sklearn.preprocessing import PolynomialFeatures

# A = np.ones((1, 4))
# B = np.ones((4, 1))

# C = np.array([[3],
#              [3],
#              [3],
#              [3]])

# print("A=", A)
# print("B=", B)
# print("C=", C)
# print("C.shape=", C.shape)
# print("Aflatten=", A.flatten())
# print("Bflatten=", B.flatten())
# print("Cflatten=", C.flatten())
# print("A*B=", np.matmul(A, B))

# X_poly = PolynomialFeatures(degree=2).fit_transform(C)

# print("X_poly=", X_poly)

# ====== Test 3 ======

from sklearn.preprocessing import PolynomialFeatures

A = np.array([[2],
             [2],
             [2],
             [2]])

B = np.array([[3],
             [3],
             [3],
             [3]])

print("A=", A)
print("B=", B)

C = np.c_[A.flatten(), B.flatten()]

print("C=", C)

X_poly = PolynomialFeatures(degree=2).fit_transform(C)

print("X_poly=", X_poly)


