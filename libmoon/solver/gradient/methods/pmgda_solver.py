import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

from libmoon.solver.gradient.methods.core.mgda_core import solve_mgda
from libmoon.solver.gradient.methods.core import BaseCore
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x, dtype=float)

def get_nn_pmgda_componets(loss_vec, pref):
    '''
        return: h_val, grad_h, J_hf
    '''
    # ensure tensor on correct device and requires grad
    if not isinstance(loss_vec, torch.Tensor):
        loss_vec = torch.tensor(loss_vec, dtype=torch.float32)
    loss_vec = loss_vec.detach().clone().to(device).requires_grad_(True)

    # constraint function is defined elsewhere in this module/project
    # call it to compute scalar h
    h = constraint(loss_vec, pref=pref)  # keep existing interface
    # compute gradient
    h_val = h.detach().cpu().item()
    h.backward()
    J_hf = loss_vec.grad.detach().cpu()
    return h_val, J_hf

def pbi(f, lamb):
    f = torch.Tensor(f).double().to(device)
    lamb = torch.Tensor(lamb).double().to(device)
    d1 = torch.dot(f, lamb) / (torch.norm(lamb) + 1e-12)
    proj = (d1.unsqueeze(0) * lamb) / (torch.norm(lamb) + 1e-12)
    d2 = torch.norm(f - proj)
    return d1, d2

def constraint(loss_arr, pref=torch.Tensor([0, 1])):
    '''
        # Just consider two types of constraints.
        # -- Type (1), the `exact' constraint.
        # -- Type (2), the `ROI' constraint.
    '''
    if type(pref) == np.ndarray:
        pref = torch.Tensor(pref).to(device)
    constraint_mtd = 'pbi'
    if constraint_mtd == 'cel':
        eps = 1e-3
        loss_arr_0 = torch.clip(loss_arr / torch.sum(loss_arr), eps)
        res = torch.sum(loss_arr_0 * torch.log(loss_arr_0 / pref)) + torch.sum(
            pref * torch.log(pref / loss_arr_0))
        d2 = res.unsqueeze(0)
    elif constraint_mtd == 'cos':
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        pref_ts = pref.to(loss_arr.device)
        d2 = (1 - cos(loss_arr.unsqueeze(0), pref_ts.unsqueeze(0)))
    else:
        _, d2 = pbi(loss_arr, pref)
        d2 = d2.unsqueeze(0)
    return d2

def solve_pmgda(Jacobian, Jacobian_h_losses, h_val, h_tol, sigma):
    '''
        Input:
        Jacobian: (n_obj, n_var) : Tensor
        grad_h: (1, n_var)
        h_val: (1,) : float
        Jhf: (m,)
        Output:
        alpha: (m,)
    '''
    grad_h = Jacobian_h_losses @ Jacobian
    Jacobian_ts = Jacobian.detach().clone().to(device)
    Jacobian_np = to_numpy(Jacobian)
    G_ts = torch.cat((Jacobian, grad_h.unsqueeze(0)), dim=0).detach()
    G_norm = torch.norm(G_ts, dim=1, keepdim=True)
    G_n = G_ts / (G_norm + 1e-4)
    GGn = (G_ts @ G_n.T).clone().cpu().numpy()
    (m, n) = Jacobian_ts.shape
    condition = h_val < h_tol
    if condition:
        mu_prime = solve_mgda(Jacobian_ts)
    try:
        A1 = -GGn
        A_tmp = -np.ones((m + 1, 1))
        A_tmp[-1][0] = 0
        A1 = np.c_[A1, A_tmp]
        b1 = np.zeros(m + 1)
        b1[-1] = -sigma * np.linalg.norm(grad_h)

        A2 = np.c_[-np.eye(m + 1), np.zeros((m + 1, 1))]
        b2 = -np.zeros(m + 1)

        A3 = np.ones((1, m + 2))
        A3[0][-1] = 0.0
        b3 = np.ones(1)

        A4 = -np.ones((1, m + 2))
        A4[0][-1] = 0.0
        b4 = -np.ones(1)

        A_all = np.concatenate((A1, A2, A3, A4), axis=0)
        b_all = np.r_[b1, b2, b3, b4]

        A_matrix = matrix(A_all)
        b_matrix = matrix(b_all)

        c = np.zeros(m + 2)
        c[-1] = 1.0
        c_matrix = matrix(c)

        sol = solvers.lp(c_matrix, A_matrix, b_matrix)
        res = np.array(sol['x']).squeeze()
        mu = res[:-1]
        mu_prime = get_pmgda_DWA_coeff(mu, Jacobian_h_losses, G_norm, m)
        return mu_prime
    except Exception:
        return solve_mgda(Jacobian_np)

def get_pmgda_DWA_coeff(mu, Jhf, G_norm, m):
    '''
        This function is to compute the coefficient of the dynamic weight adjustment.
        Please ref the Eq. (18) for the formulation in the main paper.
    '''
    mu_prime = np.zeros( m )
    for i in range( m ):
        mu_prime[i] = mu[i] / G_norm[i] + mu[m] / G_norm[m] * Jhf[i]
    return mu_prime

def get_Jhf(f_arr, pref, return_h=False):
    if not isinstance(f_arr, torch.Tensor):
        f_arr = torch.tensor(f_arr, dtype=torch.float32)
    f_var = f_arr.detach().clone().requires_grad_(True)
    h = constraint(f_var, pref=pref)
    h.backward()
    Jhf = f_var.grad.detach().clone().cpu().numpy()
    if return_h:
        return Jhf, h.detach().clone().cpu().numpy()
    return Jhf

class PMGDACore(BaseCore):
    def __init__(self, n_var, prefs):
        '''
            Input:
            n_var: int, number of variables.
            prefs: (n_prob, n_obj).
        '''
        self.core_name = 'PMGDACore'
        self.prefs = prefs
        self.n_var = n_var
        self.h_eps = 0.01
        self.sigma = 0.95

    def get_alpha(self, Jacobian, losses, idx):
        '''
            Input:
            Jacobian: (n_obj, n_var), torch.Tensor
            losses: (n_obj,), torch.Tensor
            idx: int
        '''
        # (1) get the constraint value
        losses_var = losses.detach().clone().requires_grad_(True)
        h_var = constraint(losses_var, pref=self.prefs[idx])
        h_val = h_var.detach().cpu().clone().numpy()
        h_var.backward()
        Jacobian_h_losses = losses_var.grad.detach().clone()
        # shape: (n_obj)
        alpha = solve_pmgda(Jacobian, Jacobian_h_losses, h_val, self.h_eps, self.sigma)
        return torch.Tensor(alpha).to(Jacobian.device)

class PMGDASolver(GradBaseSolver):
    # The PGMDA paper: http://arxiv.org/abs/2402.09492.
    def __init__(self, problem, prefs, step_size=1e-3, n_epoch=500, tol=1e-3,
                 sigma=0.1, h_tol=1e-3, folder_name=None):
        self.folder_name=folder_name
        self.problem = problem
        self.sigma = sigma
        self.h_tol = h_tol
        self.n_epoch = n_epoch
        self.pmgda_core = PMGDACore(n_var=problem.n_var, prefs=prefs)
        self.prefs = prefs
        self.solver_name = 'PMGDA'
        super().__init__(step_size, n_epoch, tol, self.pmgda_core)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)