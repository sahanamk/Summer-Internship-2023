'''SCIPY - SCIENTIFIC PYTHON'''
import scipy
import numpy as np
scipy.__version__

#scipy constants
from scipy import constants
constants.liter
constants.pi
dir(constants)

#scipy optimizers
from scipy.optimize import root
from math import cos

def eqn(x):
    return x+cos(x)

myroot = root(eqn,0)
myroot.x
myroot

from scipy.optimize import minimize

def eqn(x):
  return x**2 + x + 2

mymin = minimize(eqn, 0, method='BFGS')
mymin

#scipy sparse data - most items in data are 0
from scipy.sparse import csr_matrix

a=np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
csr_matrix(a)
print(csr_matrix(a))

csr_matrix(a).data

csr_matrix(a).count_nonzero()

mat=csr_matrix(a)
mat.eliminate_zeros()
print(mat)

mat=csr_matrix(a)
mat.sum_duplicates()
print(mat)

newa = csr_matrix(a).tocsc() 





















