import torch
import numpy as np
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from libmoon.problem.synthetic import ZDT1, ZDT2
from matplotlib import pyplot as plt
from libmoon.solver.gradient.methods.core.mgda_core import solve_mgda
from libmoon.solver.gradient.methods.core import BaseCore

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x, dtype=float)

'''
    MGDA solver, published in: 
    [1]. Multiple-gradient descent algorithm (MGDA) for multiobjective optimizationAlgorithme de descente à gradients multiples pour lʼoptimisation multiobjectif
    [2]. Sener, Ozan, and Vladlen Koltun. "Multi-task learning as multimnist-objective optimization." Advances in neural information processing systems 31 (2018).
'''
class MGDAUBSolver(GradBaseSolver):
    def __init__(self, problem, prefs, step_size=1e-3, n_epoch=500, tol=1e-3, folder_name=None):
        self.folder_name = folder_name
        self.mgda_core = MGDAUBCore()
        self.problem = problem
        self.prefs = prefs
        self.solver_name = 'MGDA'
        super().__init__(step_size, n_epoch, tol, self.mgda_core)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)

class MGDAUBCore(BaseCore):
    def __init__(self):
        self.core_name = 'MGDAUBCore'

    def get_alpha(self, Jacobian, losses, idx=None):
        alpha = solve_mgda(Jacobian)
        alpha = torch.Tensor(alpha)
        s = torch.sum(alpha)
        if s == 0:
            alpha = torch.ones_like(alpha) / float(alpha.numel())
        else:
            alpha = alpha / (s + 1e-12)
        return alpha

if __name__ == '__main__':
    n_prob = 10
    n_var = 10
    n_obj = 2
    problem = ZDT1(n_var=n_var)
    prefs = torch.rand(n_prob, n_obj)
    solver = MGDAUBSolver(problem=problem, prefs=prefs, step_size=1e-3, n_epoch=1000, tol=1e-6)
    x = torch.rand(n_prob, n_var)
    res = solver.solve(x_init=x)
    y_arr = res['y']
    plt.scatter(y_arr[:, 0], y_arr[:, 1])
    plt.show()

