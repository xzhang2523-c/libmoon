import torch
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from torch import Tensor
import numpy as np
from libmoon.problem.synthetic.zdt import ZDT1
from libmoon.solver.gradient import RandomCore

class CoreRandom:
    def __init__(self, args):
        self.args = args

    def get_weight(self):
        return Tensor(np.random.rand(10, 2))

class RandomSolver(GradBaseSolver):
    def __init__(self, step_size, n_epoch, tol, problem, prefs, folder_name=None):
        self.step_size = step_size
        self.n_epoch = n_epoch
        self.tol = tol
        self.problem = problem
        self.prefs = prefs
        self.random_core = RandomCore()
        self.solver_name = 'RandomSolver'
        super().__init__(step_size, n_epoch, tol, self.random_core)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)

if __name__ == '__main__':
    problem = ZDT1(n_var=10)
    pref_1d = torch.linspace(0, 1, 10)
    prefs = torch.stack((pref_1d, 1 - pref_1d), dim=1)
    solver = RandomSolver(0.1, 100, 1e-6, problem, prefs)
    x = torch.rand((10,10))
    res = solver.solve(x)
