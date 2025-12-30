"""
    [1] Xi Lin, Zhiyuan Yang, Xiaoyuan Zhang, Qingfu Zhang. Pareto Set Learning for
    Expensive Multiobjective Optimization. Advances in Neural Information Processing
    Systems (NeurIPS) , 2022.
"""
import torch
from botorch.utils.sampling import sample_simplex
from botorch.utils.multi_objective.hypervolume import Hypervolume
from .base_psl_model import ParetoSetModel
from libmoon.solver.mobo.methods.base_solver_pslmobo import PSLMOBO

torch.set_default_dtype(torch.float64)

class PSLMOBOSolver(PSLMOBO):
    def __init__(self, problem, x_init, MAX_FE, BATCH_SIZE):
        super().__init__(problem, x_init, MAX_FE, BATCH_SIZE)
        self.solver_name = 'pslmobo'
        self.coef_lcb = 0.1 # coefficient of LCB

    def _train_psl(self):
        self.psmodel = ParetoSetModel(self.n_var, self.n_obj)
        # optimizer
        optimizer = torch.optim.Adam(self.psmodel.parameters(), lr=1e-3)
        # t_step Pareto Set Learning with Gaussian Process
        dtype = torch.get_default_dtype()
        for _ in range(self.n_steps):
            self.psmodel.train()
            
            # sample n_pref_update preferences L1=1
            pref_vec = sample_simplex(d=self.n_obj, n=self.n_pref_update-self.n_obj).to(torch.float64)
            pref_vec = torch.cat((pref_vec, torch.eye(self.n_obj)),dim=0)
            pref_vec = torch.clamp(pref_vec, min=1e-6) 
            
            # get the current coressponding solutions
            x = self.psmodel(pref_vec)
            # TODO, train GPs using Gpytorch
            out = self.gps.evaluate(x.detach().cpu().numpy(), cal_std=True, cal_grad=True) 
            mean = torch.from_numpy(out['F']).to(dtype)
            std = torch.from_numpy(out['S']).to(dtype)
            mean_grad = torch.from_numpy(out['dF']).to(dtype)
            std_grad = torch.from_numpy(out['dS']).to(dtype)

            # calculate the value/grad of tch decomposition with LCB
            value = mean - self.coef_lcb * std    # n_pref_update *  n_obj  
            # n_pref_update *  n_obj * n_var   
            value_grad = mean_grad - self.coef_lcb * std_grad
            tch_idx = torch.argmax(pref_vec * (value - self.z), dim=1)  # (n_pref_update,)
            idx0 = torch.arange(len(tch_idx), dtype=torch.long)
            # select corresponding scalar weights and gradients using tuple indexing
            pref_sel = pref_vec[idx0, tch_idx].view(self.n_pref_update, 1)                 # (n_pref_update,1)
            value_grad_sel = value_grad[idx0, tch_idx, :]                                 # (n_pref_update, n_var)
            tch_grad = pref_sel * value_grad_sel                                          # (n_pref_update, n_var)
            norm = torch.norm(tch_grad, dim=1, keepdim=True)
            norm = torch.clamp(norm, min=1e-12)
            tch_grad = tch_grad / norm
             # gradient-based pareto set model update
            optimizer.zero_grad()
            self.psmodel(pref_vec).backward(tch_grad.to(self.psmodel(pref_vec).dtype))
            optimizer.step()  
            
    def _batch_selection(self, batch_size):
        # sample n_candidate preferences default:1000
        self.psmodel.eval()  # Sets the module in evaluation mode.
        dtype = torch.get_default_dtype()
        pref = sample_simplex(d=self.n_obj, n=self.n_candidate).to(dtype)
        pref = torch.clamp(pref, min=1e-6) 
        # generate correponding solutions, get the predicted mean/std
        with torch.no_grad():
            candidate_x = self.psmodel(pref).to(dtype)
            # TODO, train GPs using Gpytorch
            out = self.gps.evaluate(candidate_x.detach().cpu().numpy(), cal_std=True, cal_grad=False)
            candidate_mean, candidata_std = torch.from_numpy(out['F']).to(dtype), torch.from_numpy(out['S']).to(dtype)
          
        Y_candidate = candidate_mean - self.coef_lcb * candidata_std 
        # hv
        ref_point = torch.max(torch.cat((self.train_y_nds,Y_candidate),dim=0),axis=0).values.data
        hv = Hypervolume(ref_point=-ref_point) # botorch, maximization
        # greedy batch selection 
        best_subset_list = []
        Y_p = self.train_y_nds.clone()
        for _ in range(batch_size):        
            best_hv_value = 0
            best_subset = None
            for k in range(self.n_candidate):
                Y_comb = torch.cat((Y_p,Y_candidate[k:k+1,:]),dim=0)
                hv_value_subset = hv.compute(-Y_comb) # botorch, maximization
                if hv_value_subset > best_hv_value:
                    best_hv_value = hv_value_subset
                    best_subset = k
                    
            Y_p = torch.cat((Y_p, Y_candidate[best_subset:best_subset + 1, :]), dim=0)
            best_subset_list.append(best_subset)  
       
        # evaluate the selected n_sample solutions
        X_new = candidate_x[best_subset_list]
        return X_new