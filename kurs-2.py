import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import dia_matrix
from scipy import integrate
from scipy import linalg as LA
from scipy.optimize import minimize
from scipy import optimize
from mpmath import *


def phi(x, alpha):
    return x*np.exp((-1)*alpha*x**2)

def phi_x(r, r0):
    if r < r0:
        return -5
    else:
        return 0

def dr2_phi(x, alpha):
    return 2*alpha*x*np.exp((-1)*alpha*x**2)*(2*alpha*x**2-3)

def fin_func(alpha):
    x0 = 2
    N = len(alpha)
    for i in range(N):
        if alpha[i] <= 0:
            alpha[i] = 0.00000001
    phi_m = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            res = integrate.quad(lambda x: phi(x, alpha[i]) * phi(x, alpha[j]), 0, np.inf)[0]
            phi_m[i][j] = (res)

    Aphi_m = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            res = (-1) * integrate.quad(lambda x: phi(x, alpha[i]) * dr2_phi(x, alpha[j]) + phi(x, alpha[i]) * phi_x(x, x0), 0, np.inf)[0]
            Aphi_m[i][j] = (res)

    #вывод
    print("Альфа:")
    print(alpha)
    print("Матрица Aphi")
    print(Aphi_m)
    print("Матрица phi")
    print(phi_m)
    print()
    eigenVal, eigenVectors = LA.eig(Aphi_m, phi_m)
    print("Собственные значения")
    print(eigenVal)
    print("Собственные векторы")
    print(eigenVectors)

    return min(eigenVal)

alpha = [0.1, 0.2]
result_scipy = minimize(fin_func, alpha, method='Nelder-Mead')
print()
print(result_scipy)