import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import SGD, Adam
from tqdm import tqdm
from pymoo.indicators.hv import HV

from libmoon.util.constant import get_agg_func, solution_eps, get_hv_ref
from libmoon.util.gradient import get_moo_Jacobian_batch
from libmoon.model.simple import PFLModel
from libmoon.util.prefs import pref2angle, angle2pref, get_prefs
from libmoon.metrics.metrics import compute_lmin
from libmoon.solver.gradient.methods.core import BaseCore
criterion = torch.nn.MSELoss()


def umod_train_pfl_model(folder_name, update_idx, pfl_model, pfl_optimizer,
                    criterion, prefs, y, pfl_epoch=2000):
    prefs = torch.Tensor(prefs)
    y = torch.Tensor(y)
    prefs_angle = pref2angle(prefs)
    loss_arr = []
    for _ in range(pfl_epoch):
        y_hat = pfl_model(prefs_angle)
        loss = criterion(y_hat, y)
        loss_arr.append(loss.item())
        pfl_optimizer.zero_grad()
        loss.backward()
        pfl_optimizer.step()
    plt.figure()
    plt.plot(loss_arr)
    plt.xlabel('Iteration')
    plt.ylabel('PFL Loss')
    fig_name = os.path.join(folder_name, 'loss_{}.pdf'.format(update_idx) )
    plt.savefig(fig_name)
    print('Save to {}'.format(fig_name))
    return pfl_model

def umod_adjust_pref(prefs, pfl_model, n_adjust_epoch, main_epoch_idx, folder_name):
    prefs_angle = pref2angle(prefs).detach().clone().requires_grad_(True)
    optimizer = Adam([prefs_angle], lr=1e-3)
    lmin_arr = []
    for _ in range(n_adjust_epoch):
        y_pred = pfl_model(prefs_angle)
        lmin_val = compute_lmin(y_pred)
        optimizer.zero_grad()
        (-lmin_val).backward()   # To max the pairwise distance.
        optimizer.step()
        prefs_angle.data.clamp_(0.0, np.pi / 2)
        lmin_arr.append(float(lmin_val.detach()))
    fig_name = os.path.join(folder_name, f'lmin_{main_epoch_idx}.pdf')
    plt.figure()
    plt.plot(lmin_arr)
    plt.xlabel('Iteration')
    plt.ylabel('Lmin')
    plt.savefig(fig_name)
    plt.close()
    print('Save to {}'.format(fig_name))
    return angle2pref(prefs_angle.detach())

class AggCore(BaseCore):
    def __init__(self, prefs, agg_name='ls'):
        self.core_name = 'AggCore'
        self.agg_name = agg_name
        self.prefs = prefs

    def get_alpha(self, Jacobian, losses, idx=None):
        if self.agg_name == 'ls':
            return torch.Tensor(self.prefs[idx]) if idx is not None else torch.Tensor(self.prefs)
        else:
            raise NotImplementedError(f'agg_name "{self.agg_name}" is not implemented in AggCore.')

class RandomCore(BaseCore):
    def __init__(self):
        self.core_name = 'RandomCore'

    def get_alpha(self, Jacobian, losses, idx=None):
        n_obj = len(losses)
        alpha = torch.rand(n_obj)
        alpha = alpha / torch.sum(alpha)
        return alpha

class GradBaseSolver:
    def __init__(self, step_size, epoch, tol, core_solver, verbose=False):
        self.step_size = step_size
        self.epoch = epoch
        self.tol = tol
        self.core_solver = core_solver
        self.is_agg = (getattr(self.core_solver, "core_name", "") == "AggCore")
        self.solver_name = getattr(self.core_solver, "core_name", "GradBaseSolver")
        self.agg_name = getattr(self.core_solver, "agg_name", None)
        self.verbose = verbose


    def solve(self, problem, x, prefs):
        '''
            :param problem:
            :param x:
            :return:
                is a dict with keys: x, y.
        '''
        if self.solver_name == 'UMOD':
            self.pfl_model = PFLModel(n_obj=problem.n_obj)
            self.pfl_optimizer = torch.optim.Adam(self.pfl_model.parameters(), lr=1e-3)

        self.n_prob, self.n_obj = prefs.shape[0], prefs.shape[1]
        xs_var = x.detach().clone().requires_grad_(True)
        optimizer = Adam([xs_var], lr=self.step_size)
        ind = HV(ref_point=get_hv_ref(problem.problem_name))
        hv_arr, y_arr = [], []
        # For UMOD solver, we need to store (pref, y) pairs.
        pref_y_pairs = []

        for epoch_idx in tqdm(range(self.epoch)):
            fs_var = problem.evaluate(xs_var)
            y_np = fs_var.detach().numpy()
            y_arr.append(y_np)
            hv_arr.append(ind.do(y_np))

            Jacobian_array = get_moo_Jacobian_batch(xs_var, fs_var, self.n_obj)
            y_detach = fs_var.detach()
            optimizer.zero_grad()
            
            if self.is_agg or self.solver_name == 'UMOD':
                print('agg_name', self.agg_name)
                agg_func = get_agg_func(self.agg_name)
                agg_val = agg_func(fs_var, torch.Tensor(prefs).to(fs_var.device))
                agg_val.sum().backward()
            else:
                core_name = getattr(self.core_solver, 'core_name', '')
                if core_name in ['EPOCore', 'MGDAUBCore', 'PMGDACore', 'RandomCore']:
                    alpha_array = torch.stack(
                        [self.core_solver.get_alpha(Jacobian_array[idx], y_detach[idx], idx) for idx in
                         range(self.n_prob)])
                elif core_name in ['PMTLCore', 'MOOSVGDCore', 'GradHVCore']:
                    if core_name == 'GradHVCore':
                        alpha_array = self.core_solver.get_alpha_array(y_detach)
                    elif core_name == 'PMTLCore':
                        alpha_array = self.core_solver.get_alpha_array(Jacobian_array, y_np, epoch_idx)
                    elif core_name == 'MOOSVGDCore':
                        alpha_array = self.core_solver.get_alpha_array(Jacobian_array, y_detach)
                    else:
                        assert False, 'Unknown core_name:{}'.format(core_name)
                else:
                    assert False, 'Unknown core_name'
                torch.sum(alpha_array * fs_var).backward()
            optimizer.step()
            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps,
                                     torch.Tensor(problem.ubound) - solution_eps)
            if getattr(problem, 'problem_name', None) in ['MOKL']:
                x.data = torch.clamp(x.data, min=0)
                x.data = x.data / torch.sum(x.data, dim=1, keepdim=True)

            if self.solver_name == 'UMOD':
                if epoch_idx % self.pfl_train_epoch == 0 and epoch_idx != 0:
                    pref_y_pairs.append((prefs, y_np))
                    print('Pair len: {}'.format(len(pref_y_pairs)) )
                    prefs_np = prefs.detach().numpy()
                    plt.scatter(prefs_np[:,0], prefs_np[:,1])
                    plt.scatter(y_np[:,0], y_np[:,1])
                    prefs_all = torch.cat([torch.Tensor(pair[0]) for pair in pref_y_pairs], axis=0)
                    y_all = torch.cat([torch.Tensor(pair[1]) for pair in pref_y_pairs], axis=0)
                    umod_train_pfl_model(
                        folder_name=getattr(self, 'folder_name', '.'),
                        update_idx=epoch_idx,
                        pfl_model=self.pfl_model,
                        pfl_optimizer=self.pfl_optimizer,
                        criterion=criterion,
                        prefs=prefs_all,
                        y=y_all
                    )     # Use all historical data to train the model.

                    prefs_new = umod_adjust_pref(prefs, pfl_model=self.pfl_model, n_adjust_epoch=getattr(self, 'pref_adjust_epoch', 200),
                                                 main_epoch_idx=epoch_idx, folder_name=getattr(self, 'folder_name', '.'))
                    pref_test = get_prefs(n_prob=100, dtype='Tensor')
                    y_test = self.pfl_model(pref2angle(pref_test))
                    y_test_np = y_test.detach().numpy()
                    for (pp, yy) in zip(prefs_np, y_np):
                        plt.plot([pp[0], yy[0]], [pp[1], yy[1]], color='grey', linestyle='dashed')
                    plt_umod = False
                    if plt_umod:
                        plt.scatter(y_test_np[:, 0], y_test_np[:, 1])
                        plt.scatter(prefs_new[:,0], prefs_new[:,1], label='New prefs')
                        plt.legend()
                        plt.show()
                    prefs = prefs_new
        res = {
            'x': x.detach().numpy(),
            'y': y_np,
            'hv_history': hv_arr,
            'y_history': y_arr
        }
        return res

class GradAggSolver(GradBaseSolver):
    def __init__(self, problem, prefs, step_size=1e-3, n_epoch=500, tol=1e-3,
                 agg_name='ls', folder_name=None):
        self.folder_name = folder_name
        self.problem = problem
        self.prefs = prefs
        self.solver_name = 'GradAgg'
        self.core_solver = AggCore(prefs, agg_name)
        super().__init__(step_size, n_epoch, tol, core_solver=self.core_solver)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)