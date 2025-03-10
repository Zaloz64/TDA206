import numpy as np
import cvxpy as cp
import math

# adj_mat = np.array([[   0.  , 2., 1000.  ,  2. , 1000.  ,  2.  ,  2. ,1000.  ,  2.  ,  2.],
#  [ 2.  ,  0. ,1000.  ,  2., 1000.  ,  2.  ,  2. ,1000.  ,  2.  ,  2.],
#  [1000. , 1000.  ,  0. ,1000. ,   2. ,1000., 1000. ,   2. ,1000., 1000.],
#  [   2.  ,  2. ,1000.  ,  0. ,1000.  ,  2. ,   2., 1000.  ,  2.  ,  2.],
#  [1000., 1000.  ,  2. ,1000. ,   0., 1000. ,1000.  ,  2. ,1000. ,1000.],
#  [   2.  ,  2., 1000.  ,  2., 1000. ,   0. ,   2., 1000. ,   2. ,   2.],
#  [   2.  ,  2. ,1000.  ,  2. ,1000. ,   2. ,   0., 1000. ,   2. ,   2.],
#  [1000.  ,1000.,    2. ,1000.,    2., 1000., 1000.,    0., 1000., 1000.],
#  [   2. ,   2. ,1000. ,   2. ,1000. ,   2. ,   2., 1000.,    0.,    2.],
#  [   2. ,   2. ,1000. ,   2. ,1000. ,   2. ,   2., 1000.,    2.,    0.]])

adj_mat = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

n = adj_mat.shape[0]
X = cp.Variable((n, n), symmetric = True)
ones = np.ones((n,n))

cons = [X >> 0]
for i in range(n):
    cons.append(X[i][i] == 1)


obj = cp.Maximize(cp.trace(adj_mat @ (ones-X)/4))
problem = cp.Problem(obj, cons)

problem.solve()
print("The optimal value is", problem.value)
print("A solution X is")
print(X.value)


Xnew = X.value
eigs = np.linalg.eigh(Xnew)[0]
if np.min(eigs) < 0:
  Xnew = Xnew + (1.00001 * abs(min(eigs)) * np.identity(n))
elif np.min(eigs) == 0:
  Xnew = Xnew + 0.0000001 * np.identity(n)
U = np.linalg.cholesky(Xnew)
U = U[:, :-1]

print(U)

plane = np.random.rand(n-1,1)
solution = []
for u in U:
    if np.dot(plane.T, u) >= 0:
      solution.append(1)
    else:
      solution.append(-1)
  
print(solution)




func_values = []
for i in range(n):
    for j in range(i+1, n):
        temp = X.value[i][j]
        # Clamp temp to be within [-1, 1]
        temp = max(-1.0, min(1.0, temp))
        func_values.append(2*math.acos(temp) / (math.pi *(1-temp)))


print("approx_ratio", min(func_values))