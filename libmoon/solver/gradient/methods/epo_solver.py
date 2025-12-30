import numpy as np
import cvxpy as cp
import cvxopt
import torch
import warnings
warnings.filterwarnings("ignore")
from libmoon.problem.synthetic.zdt import ZDT1
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from libmoon.solver.gradient.methods.core import BaseCore
from matplotlib import pyplot as plt

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x, dtype=float)

def mu(rl, normed=False):
    # Modified by Xiaoyuan to handle negative issue.
    # if len(np.where(rl < 0)[0]):
    #     raise ValueError(f"rl<0 \n rl={rl}")
    #     return None
    rl = to_numpy(rl)
    rl = np.clip(rl, 0.0, np.inf)
    m = len(rl)
    if m == 0:
        return 0.0
    if not normed:
        s = rl.sum()
        if s == 0:
            return 0.0
        l_hat = rl / s
    else:
        l_hat = rl
    eps = np.finfo(float).eps
    l_hat = l_hat[l_hat > eps]
    if l_hat.size == 0:
        return 0.0
    return float(np.sum(l_hat * np.log(l_hat * m)))

def adjustments(l, r=1):
    l = to_numpy(l)
    rl = r * l
    if rl.sum() == 0:
        return rl, 0.0, np.zeros_like(rl)
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    a = r * (np.log(np.clip(l_hat * len(rl), 1e-8, np.inf)) - mu_rl)
    return rl, mu_rl, a

class EPO_LP(object):
    # Paper:
    # https://proceedings.mlr.press/v119/mahapatra20a.html,
    # https://arxiv.org/abs/2010.06313
    def __init__(self, m, n, r, eps=1e-4):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.r = r
        self.eps = eps

        self.last_move = None
        self.a = cp.Parameter(m)  # Adjustments
        self.C = cp.Parameter((m, m))  # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)  # d_bal^TG
        self.rhs = cp.Parameter(m)  # RHS of constraints for balancing
        self.alpha = cp.Variable(m)  # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)  # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        cons_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        cons_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        
        self.prob_dom = cp.Problem(obj_dom, cons_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, cons_rel)  # LP dominance
        self.gamma = 0  # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0  # Stores the latest non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        rl, self.mu_rl, a = adjustments(l, r)
        self.a.value = a
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        try:
            if self.mu_rl > self.eps:
                J = self.Ca.value > 0
                if J.any():
                    J_star_idx = int(np.argmax(rl))
                    rhs = self.Ca.value.copy()
                    rhs[J] = -1e6
                    rhs[J_star_idx] = 0
                    self.rhs.value = rhs
                else:
                    self.rhs.value = np.zeros_like(self.Ca.value)
                self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
                self.last_move = "bal"
            else:
                if relax:
                    self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
                else:
                    self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
                self.last_move = "dom"

            alpha = self.alpha.value
            if alpha is None or np.any(np.isnan(alpha)):
                raise ValueError("Invalid alpha from solver")
            return alpha
        except Exception:
            losses = to_numpy(l)
            if losses.sum() == 0:
                return np.ones(self.m) / float(self.m)
            return losses / losses.sum()

def solve_epo(grad_arr, losses, pref, epo_lp):
    '''
        input: grad_arr: (m,n).
        losses : (m,).
        pref: (m,) inv.
        return : gw: (n,). alpha: (m,)
    '''
    pref = to_numpy(pref)
    G = to_numpy(grad_arr)
    losses_np = to_numpy(losses).squeeze()
    GG = G @ G.T
    alpha = epo_lp.get_alpha(losses_np, G=GG, C=True)
    if alpha is None or np.any(np.isnan(alpha)):
        # fallback to pref proportion
        s = pref.sum()
        if s == 0:
            alpha = np.ones_like(pref) / float(len(pref))
        else:
            alpha = pref / s
    gw = alpha @ G
    return torch.Tensor(gw), alpha

class EPOSolver(GradBaseSolver):
    def __init__(self, problem, prefs, step_size=1e-3, n_epoch=500, tol=1e-3, folder_name=None):
        self.folder_name = folder_name
        self.solver_name = 'EPO'
        self.problem = problem
        self.prefs = prefs
        self.epo_core = EPOCore(n_var=problem.n_var, prefs=prefs)
        super().__init__(step_size, n_epoch, tol, self.epo_core)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)


class EPOCore(BaseCore):
    def __init__(self, n_var, prefs):
        '''
            Input:
            n_var: int, number of variables.
            prefs: (n_prob, n_obj).
        '''
        self.core_name = 'EPOCore'
        self.prefs = prefs
        self.n_prob, self.n_obj = prefs.shape[0], prefs.shape[1]
        self.n_var = n_var
        prefs_np = to_numpy(prefs)
        self.epo_lp_arr = [EPO_LP(m=self.n_obj, n = self.n_var, r=1/pref) for pref in prefs_np]

    def get_alpha(self, Jacobian, losses, idx):
        _, alpha = solve_epo(Jacobian, losses, self.prefs[idx], self.epo_lp_arr[idx])
        return torch.Tensor(alpha)


if __name__ == '__main__':
    n_obj, n_var, n_prob = 2, 10, 8
    # prefs = np.random.rand(n_prob, n_obj)
    pref_1d = np.linspace(0.1, 0.9, n_prob)
    prefs = np.c_[pref_1d, 1 - pref_1d]
    problem = ZDT1(n_var=n_var)
    solver = EPOSolver(step_size=1e-2, n_iter=1000, tol=1e-3, problem=problem, prefs=prefs)
    x0 = torch.rand(n_prob, n_var)
    res = solver.solve(x=x0)
    y_arr = res['y']
    plt.scatter(y_arr[:, 0], y_arr[:, 1], color='black')
    plt.show()
