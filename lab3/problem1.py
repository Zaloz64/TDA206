import cvxpy as cp
import numpy as np

x = cp.Variable((5, 1), nonneg=True)
c_primal = np.array([4, -2, 5, 6, 7])
const_primal = [
    2*x[0] + 2*x[1] - 4*x[2] + 4*x[3] + 8*x[4] <= 6,
    2*x[0] + x[1] - 2*x[2] - x[3] - 3*x[4] >= -1,
    5*x[0] - 2*x[1] + 4*x[2] + 4*x[3] + 2*x[4] == 5,
    2*x[0] - 2*x[1] + 5*x[2] + 3*x[3] + x[4] <= 4,
]

objective_primal = cp.Maximize(c_primal @ x)

problem_primal = cp.Problem(objective_primal, const_primal)
solution_primal = problem_primal.solve()


y = cp.Variable((4, 1))
c_dual = np.array([6, -1, 5, 4])

const_dual = [
    2*y[0] + 2*y[1] + 5*y[2] + 2*y[3] >= 4,
    2*y[0] + y[1] - 2*y[2] - 2*y[3] >= -2,
    (-4)*y[0] - 2*y[1] + 4*y[2] + 5*y[3] >= 5,
    4*y[0] - y[1] + 4*y[2] + 3*y[3] >= 6,
    8*y[0] - 3*y[1] + 2*y[2] + y[3] >= 7,
    y[0]>=0, y[1] <= 0, y[3] >= 0
]

objective_dual = cp.Minimize(c_dual @ y)

problem_dual = cp.Problem(objective_dual, const_dual)
solution_dual = problem_dual.solve()


const = np.array([[2,2,-4, 4,8], [2,1,-2,-1,-3], [5,-2,4,4,2], [2,-2,5,3,1]]) # Constraints as matrix to make following implementation easier
const_dual_M = const.transpose()

# Check Complementary Slackness for Primal
for i in range(len(x.value)):
    if x.value[i] == 0:
        continue
    else:
        print(f"{const_dual_M[i].dot(y.value)} should be equal to {c_primal[i]}")

# Check Complementary Slackness for Dual
for i in range(len(y.value)):
    if y.value[i] == 0:
        continue
    else:
        print(f"{const[i].dot(x.value)} should be equal to {c_dual[i]}")
