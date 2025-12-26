import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from scipy.sparse import random
from training import experiment, get_accuracy, DGC, DGC2
from utils import CustomDistribution, convert_scipy_to_tensor
from models import Net
import torch
import time
import tracemalloc
from math import inf
from data_loading import get_block_matrices2

# Ensure all tensors are transferred to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HyperparameterTuner:
    def __init__(self, num_values: int):
        self.best_accuracy1 = -np.inf
        self.best_accuracy_DGC = [-np.inf] * num_values
        self.best_accuracy_DGC2 = [-np.inf] * num_values
        self.best_accuracy_FGC = [-np.inf] * (num_values + 1)
        self.min_accuracy_DGC = float('inf')
        self.min_accuracy_DGC2 = float('inf')
        self.min_accuracy_FGC = float('inf')
        self.best_C_0 = None
        self.C_return = None
        self.time_DGC = float('inf')
        self.time_DGC2 = float('inf')
        self.time_FGC = float('inf')
        self.X_old = None
        self.X_old_store = None
        self.peak_memory_DGC = 0
        self.peak_memory_DGC2 = 0
        self.peak_memory_FGC = 0
        self.X_coarse = None
        self.X_coarse_tmp = None
        self.cs_ret = None
        self.adj = None
        # Store convergence histories for plotting (best runs per timestep)
        self.loss_history_dgc = [None] * num_values
        self.loss_history_dgc2 = [None] * num_values

    def objective_base_FGC(self, params: dict, k: int, n: int, p: int, theta: torch.Tensor, X: torch.Tensor, labels: np.ndarray, num_classes: int, adj: torch.Tensor, edge_index_original: torch.Tensor) -> float:
        lambda_param = params['lambda_param']
        beta_param = params['beta_param']
        alpha_param = params['alpha_param']
        gamma_param = params['gamma_param']
        av = []
        start_time = time.time()

        # Custom distribution for generating random sparse matrices
        temp = CustomDistribution(seed=1)
        temp2 = temp()  # Get a frozen version of the distribution
        X_t_0 = None

        for _ in range(1):  # Single iteration
            X_tilde = random(k, n, density=0.15, random_state=1, data_rvs=temp2.rvs)
            C = random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)

            # Convert sparse matrices to CUDA tensors
            X_t_0, C_0 = experiment(
                lambda_param,
                beta_param,
                alpha_param,
                gamma_param,
                convert_scipy_to_tensor(C),
                convert_scipy_to_tensor(X_tilde),
                theta.to(device),
                X.to(device),
                k,
                p
            )
            L = theta.to(device)
            # acc = get_accuracy2(edge_index_original.to(device), C_0, L, X.to(device))
            acc = get_accuracy(C_0, L, X_t_0.to(device), labels, num_classes, k, X.to(device), adj.to(device), Net)
            av.append(acc)

        avg_accuracy = np.mean(av)
        print(f"Average accuracy = {avg_accuracy*100:.2f}% with params: {params}")
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for base FGC is: {elapsed_time:.2f} seconds")

        if avg_accuracy > self.best_accuracy1:
            self.best_accuracy1 = avg_accuracy
            self.best_C_0 = C_0
            print(f"New best accuracy = {self.best_accuracy1*100:.2f}%")
            self.X_old = X_t_0
            self.X_old_store = X_t_0
            self.X_coarse = X_t_0

        return -avg_accuracy

    def tune_hyperparameters(self, k: int, n: int, p: int, theta: torch.Tensor, X: torch.Tensor, labels: np.ndarray, num_classes: int, adj: torch.Tensor, edge_index_original: torch.Tensor) -> tuple:
        search_space = {
            'lambda_param': hp.loguniform('lambda_param', np.log(0.1), np.log(100)),
            'beta_param': hp.loguniform('beta_param', np.log(0.1), np.log(100)),
            'alpha_param': hp.loguniform('alpha_param', np.log(0.1), np.log(100)),
            'gamma_param': hp.loguniform('gamma_param', np.log(0.1), np.log(100))
        }

        trials = Trials()
        print('Checkpoint 7')
        best = fmin(
            fn=lambda params: self.objective_base_FGC(params, k, n, p, theta, X, labels, num_classes, adj, edge_index_original),
            space=search_space,
            algo=tpe.suggest,
            max_evals=25, # Reduced for faster debugging, can be increased
            trials=trials
        )
        print("Best hyperparameters:", best)
        print(f"Best accuracy achieved: {self.best_accuracy1*100:.2f}%")
        return best, self.best_accuracy1, self.best_C_0

    def objective_FGC(
    self, 
    params: dict, 
    i: int, 
    k: int, 
    n: int, 
    p: int, 
    theta: torch.Tensor, 
    X: torch.Tensor, 
    labels: np.ndarray, 
    num_classes: int, 
    adj: torch.Tensor, 
    edge_index_original: torch.Tensor
) -> float:
        lambda_param = params['lambda_param']
        beta_param = params['beta_param']
        alpha_param = params['alpha_param']
        gamma_param = params['gamma_param']
        av = []
        start_time = time.time()
        
        temp = CustomDistribution(seed=1)
        temp2 = temp()  # Get a frozen version of the distribution
        
        peak_memory = None
        for _ in range(1):  # Single iteration
            X_tilde = random(k, n, density=0.15, random_state=1, data_rvs=temp2.rvs)
            C = random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)

            # Memory tracing for GPU/CPU
            tracemalloc.start()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            X_t_0, C_0 = experiment(
                lambda_param, 
                beta_param, 
                alpha_param, 
                gamma_param, 
                convert_scipy_to_tensor(C), 
                convert_scipy_to_tensor(X_tilde), 
                theta.to(device), 
                X.to(device), 
                k, 
                p
            )

            # Collect peak memory usage
            current, peak_cpu = tracemalloc.get_traced_memory()
            if torch.cuda.is_available():
                peak_gpu = torch.cuda.max_memory_allocated(device=device) / 1024  # Convert to KB
            else:
                peak_gpu = 0
            tracemalloc.stop()
            peak_memory = peak_cpu / 1024 + peak_gpu  # Combine CPU and GPU memory usage in KB

            L = theta.to(device)
            acc = get_accuracy(C_0, L, X_t_0.to(device), labels, num_classes, k, X.to(device), adj.to(device), Net)
            av.append(acc)

        avg_accuracy = np.mean(av)
        print(f"Average accuracy = {avg_accuracy*100:.2f}% with params: {params}")
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for FGC for iteration number {i+1} is: {elapsed_time:.2f} seconds")
        print(f"Peak Memory FGC: {peak_memory:.2f} KB")
        
        # Update best results
        if avg_accuracy > self.best_accuracy_FGC[i]:
            self.best_accuracy_FGC[i] = avg_accuracy
            self.best_C_0 = C_0
            self.time_FGC = elapsed_time
            self.peak_memory_FGC = peak_memory /1024 # converted to MB
            print(f"New best accuracy = {self.best_accuracy_FGC[i]*100:.2f}%")
        
        # Update minimum accuracy
        if avg_accuracy < self.min_accuracy_FGC:
            self.min_accuracy_FGC = avg_accuracy

        return -avg_accuracy

    def tune_hyperparameters_FGC(
        self, 
        max_iter: int, 
        i: int, 
        k: int, 
        n: int, 
        p: int, 
        theta: torch.Tensor, 
        X: torch.Tensor, 
        labels: np.ndarray, 
        num_classes: int, 
        adj: torch.Tensor, 
        edge_index_original: torch.Tensor
    ) -> tuple:
        search_space = {
            'lambda_param': hp.loguniform('lambda_param', np.log(0.1), np.log(100)),
            'beta_param': hp.loguniform('beta_param', np.log(0.1), np.log(100)),
            'alpha_param': hp.loguniform('alpha_param', np.log(0.1), np.log(100)),
            'gamma_param': hp.loguniform('gamma_param', np.log(0.1), np.log(100))
        }

        trials = Trials()
        best = fmin(
            fn=lambda params: self.objective_FGC(
                params, 
                i, 
                k, 
                n, 
                p, 
                theta, 
                X, 
                labels, 
                num_classes, 
                adj, 
                edge_index_original
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=2, # Reduced for faster debugging, can be increased
            trials=trials
        )

        print("Best hyperparameters:", best)
        print(f"Best accuracy achieved: {self.best_accuracy_FGC[i]*100:.2f}%")
        # **FIX**: Return the 'best' dictionary
        return best, self.best_accuracy_FGC[i], self.min_accuracy_FGC, self.best_C_0, self.time_FGC, self.peak_memory_FGC
            
    def objective_DGC(
        self,
        i: int,
        params1: dict,
        A_new: torch.Tensor,
        A_old: torch.Tensor,
        C_old: torch.Tensor,
        size_new: int,
        size_old: int,
        new_nodes: int,
        theta_new: torch.Tensor,
        labels: np.ndarray,
        num_classes: int,
        k: int,
        X_new: torch.Tensor,
        edge_index_original: torch.Tensor
    ) -> float:
        alpha_param = params1['alpha_param']
        beta_param = params1['beta_param']
        av = []
        start_time = time.time()
        
        temp = CustomDistribution(seed=1)
        temp2 = temp()  # Get a frozen version of the distribution
        
        if new_nodes <= 0:
            raise ValueError("new_nodes must be greater than 0")

        peak_memory = None
        status = True
        X_t_0 = None

        for _ in range(1):  # Single iteration
            sparse_delta_C = random(new_nodes, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
            dense_array = sparse_delta_C.toarray()
            delta_C = torch.tensor(dense_array, dtype=torch.float32, requires_grad=True)

            X_t_0, C_0, peak_memory, status, loss_history = DGC(
                alpha_param,
                beta_param,
                delta_C.to(device),
                A_new.to(device),
                A_old.to(device),
                C_old.to(device),
                size_new,
                size_old,
                new_nodes,
                X_new.to(device),
                self.X_old_store.to(device),
                k
            )

            if not status:
                break

            L = theta_new.to(device)
            acc = get_accuracy(C_0, L, X_t_0.to(device), labels, num_classes, k, X_new.to(device), A_new.to(device), Net)
            av.append(acc)

        if status:
            avg_accuracy = np.mean(av)
            print(f"Average accuracy = {avg_accuracy * 100:.2f}% with params: {params1}")
            elapsed_time = time.time() - start_time
            print(f"Elapsed time for DGC: {elapsed_time:.2f} seconds")
            print(f"Peak Memory DGC: {peak_memory:.2f} B")

            # Update best results
            if avg_accuracy > self.best_accuracy_DGC[i]:
                self.best_accuracy_DGC[i] = avg_accuracy
                self.C_return = C_0
                self.time_DGC = elapsed_time
                self.X_old = X_t_0
                self.peak_memory_DGC = peak_memory / (1024*1024) # converted to MB
                self.loss_history_dgc[i] = loss_history
                print(f"New best accuracy = {self.best_accuracy_DGC[i] * 100:.2f}%")
            
            # Update minimum accuracy
            if avg_accuracy < self.min_accuracy_DGC:
                self.min_accuracy_DGC = avg_accuracy
        else:
            avg_accuracy = 0

        return -avg_accuracy

    def tune_hyperparameters_DGC(
        self,
        max_iter: int,
        i: int,
        A_new: torch.Tensor,
        A_old: torch.Tensor,
        C_old: torch.Tensor,
        size_new: int,
        size_old: int,
        new_nodes: int,
        theta_new: torch.Tensor,
        labels: np.ndarray,
        num_classes: int,
        k: int,
        X_new: torch.Tensor,
        edge_index_original: torch.Tensor
    ) -> tuple:
        search_space1 = {
            'alpha_param': hp.loguniform('alpha_param', np.log(0.1), np.log(100)),
            'beta_param': hp.loguniform('beta_param', np.log(0.1), np.log(100))
        }

        self.X_old_store = self.X_old
        trials = Trials()

        best = fmin(
            fn=lambda params1: self.objective_DGC(
                i,
                params1,
                A_new,
                A_old,
                C_old,
                size_new,
                size_old,
                new_nodes,
                theta_new,
                labels,
                num_classes,
                k,
                X_new,
                edge_index_original
            ),
            space=search_space1,
            algo=tpe.suggest,
            max_evals=20, # Reduced for faster debugging, can be increased
            trials=trials
        )

        print("Best hyperparameters:", best)
        print(f"Best accuracy achieved: {self.best_accuracy_DGC[i] * 100:.2f}%")
        # **FIX**: Return the 'best' dictionary
        return best, self.best_accuracy_DGC[i], self.min_accuracy_DGC, self.C_return, self.time_DGC, self.peak_memory_DGC

#  -------------------------------------------------------------------Case2----------------------------------------------------------------------

    def objective_DGC2(
    self,
    i: int,
    params1: dict,
    A_new: torch.Tensor,
    A_old: torch.Tensor,
    C_old: torch.Tensor,
    size_new: int,
    size_old: int,
    new_nodes: int,
    theta_new: torch.Tensor,
    labels: np.ndarray,
    num_classes: int,
    k: int,
    X_new: torch.Tensor,
    X_coarse_old: torch.Tensor,
    edge_index_original: torch.Tensor,
    A_c_surv: torch.Tensor # Added parameter
) -> float:
        # Extract hyperparameters
        alpha1_param = params1['alpha1_param']
        beta1_param = params1['beta1_param']
        omega1_param = params1['omega1_param']
        alpha2_param = params1['alpha2_param']
        
        av = []
        start_time = time.time()

        temp = CustomDistribution(seed=1)
        temp2 = temp()  # Get a frozen version of the distribution

        if new_nodes <= 0:
            raise ValueError("new_nodes must be greater than 0")

        peak_memory = None
        status = True
        cs = None
        X_t_0 = None

        for _ in range(1):  # Single iteration
            delta_CN_sparse = random(new_nodes, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
            size_C_old = C_old.size()
            delta_CR_sparse = random(size_C_old[0], k, density=0.15, random_state=1, data_rvs=temp2.rvs)

            delta_CN = torch.tensor(delta_CN_sparse.toarray(), dtype=torch.float32, requires_grad=True)
            delta_CR = torch.tensor(delta_CR_sparse.toarray(), dtype=torch.float32, requires_grad=True)
            
            # Perform DGC2 operation
            X_t_0, C_0, peak_memory, status, cs, loss_history = DGC2(
                alpha1_param, beta1_param, omega1_param,
                alpha2_param,
                delta_CN.to(device),
                delta_CR.to(device),
                A_new.to(device), A_old.to(device), C_old.to(device),
                size_new, size_old, new_nodes,
                X_new.to(device), k, X_coarse_old.to(device),
                A_c_surv.to(device) if A_c_surv is not None else None # Pass A_c_surv
            )

            if not status:
                break

            L = theta_new.to(device)
            acc = get_accuracy(C_0, L, X_t_0.to(device), labels, num_classes, k, X_new.to(device), A_new.to(device), Net)
            av.append(acc)

        if status:
            avg_accuracy = np.mean(av)
            print(f"Average accuracy = {avg_accuracy * 100:.2f}% with params: {params1}")
            elapsed_time = time.time() - start_time
            print(f"Elapsed time for DGC2: {elapsed_time:.2f} seconds")
            print(f"Peak Memory DGC2: {peak_memory:.2f} B")

            # Update best results
            if avg_accuracy > self.best_accuracy_DGC2[i]:
                self.best_accuracy_DGC2[i] = avg_accuracy
                self.cs_ret = cs
                self.C_return = C_0
                self.time_DGC2 = elapsed_time
                self.X_coarse_tmp = X_t_0
                self.peak_memory_DGC2 = peak_memory / (1024*1024) # converted to MB
                self.loss_history_dgc2[i] = loss_history
                print(f"New best accuracy = {self.best_accuracy_DGC2[i] * 100:.2f}%")
            
            # Update minimum accuracy
            if avg_accuracy < self.min_accuracy_DGC2:
                self.min_accuracy_DGC2 = avg_accuracy
        else:
            avg_accuracy = 0

        return -avg_accuracy

    def tune_hyperparameters_DGC2(
        self,
        max_iter: int,
        i: int,
        A_new: torch.Tensor,
        A_old: torch.Tensor,
        C_old: torch.Tensor,
        size_new: int,
        size_old: int,
        new_nodes: int,
        theta_new: torch.Tensor,
        labels: np.ndarray,
        num_classes: int,
        k: int,
        X_new: torch.Tensor,
        edge_index_original: torch.Tensor,
        A_c_surv: torch.Tensor # Added parameter
    ) -> tuple:
        search_space1 = {
            'alpha1_param': hp.loguniform('alpha1_param', np.log(0.1), np.log(100)),
            'beta1_param': hp.loguniform('beta1_param', np.log(0.1), np.log(100)),
            'omega1_param': hp.loguniform('omega1_param', np.log(0.1), np.log(100)),
            'alpha2_param': hp.loguniform('alpha2_param', np.log(0.1), np.log(100))
        }

        trials = Trials()

        # Hyperparameter optimization
        best = fmin(
            fn=lambda params1: self.objective_DGC2(
                i, params1, A_new, A_old, C_old,
                size_new, size_old, new_nodes,
                theta_new, labels, num_classes,
                k, X_new, self.X_coarse, edge_index_original,
                A_c_surv # Pass down
            ),
            space=search_space1,
            algo=tpe.suggest,
            max_evals=20, # Reduced for faster debugging, can be increased
            trials=trials
        )

        # Update coarse graph representation
        self.X_coarse = self.X_coarse_tmp

        print("Best hyperparameters:", best)
        print(f"Best accuracy achieved: {self.best_accuracy_DGC2[i] * 100:.2f}%")

        # Compute adjacency difference
        M1, _, _, _ = get_block_matrices2(A_old, A_new, size_old, new_nodes)
        diff = M1 - A_old
        self.adj = torch.sum(torch.abs(diff))

        # **FIX**: Return the 'best' dictionary
        return best, self.best_accuracy_DGC2[i],self.min_accuracy_DGC2,self.C_return,self.time_DGC2,self.peak_memory_DGC2,self.cs_ret,self.adj
