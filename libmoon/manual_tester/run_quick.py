from libmoon.util.synthetic import synthetic_init
from libmoon.util.prefs import get_prefs
from libmoon.util.problems import get_problem
from libmoon.solver.gradient.methods import EPOSolver
from libmoon.solver.psl.core_psl import BasePSLSolver
from libmoon.util import get_problem
from torch import Tensor

if __name__ == '__main__':
    problem = get_problem(problem_name='ZDT1')
    prefs = get_prefs(n_prob=100, n_obj=problem.n_obj, clip_eps=1e-2)
    solver = BasePSLSolver(problem, solver_name='agg_ls')
    model, _ = solver.solve()
    eval_y = problem.evaluate(model(Tensor(prefs).cuda()))
