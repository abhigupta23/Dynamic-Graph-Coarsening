from data_loading import preprocess_data, get_laplacian
from utils import CustomDistribution, align_graph_components, prune_supernodes
from training import experiment, get_accuracy, get_accuracy_without_coarsening
from hyperparameter_tuning import HyperparameterTuner
from torch_geometric.utils import to_dense_adj
import numpy as np
import torch
from models import Net
from as733_data_loader import AS733DynamicGraph
from sklearn.metrics import normalized_mutual_info_score
import os
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# --- Efficient Metric Computation Functions ---
def compute_ree_vectorized(theta, C, top_k=100):
    """
    Compute Relative Eigenvalue Error (REE) efficiently using vectorized operations.
    Compares top-k eigenvalues of original Laplacian vs coarsened Laplacian.
    """
    with torch.no_grad():
        # Compute eigenvalues of original Laplacian (sorted by magnitude)
        eigen_orig = torch.linalg.eigvalsh(theta)  # Real symmetric, use eigvalsh for efficiency
        s = eigen_orig[-top_k:]  # Top-k largest eigenvalues
        
        # Compute coarsened Laplacian eigenvalues
        L_coarse = C.T @ theta @ C
        eigen_coarse = torch.linalg.eigvalsh(L_coarse)
        z = eigen_coarse[-top_k:]  # Top-k largest eigenvalues
        
        # Vectorized REE computation: mean of |z_i - s_i| / |s_i|
        # Avoid division by zero
        s_safe = torch.clamp(torch.abs(s), min=1e-10)
        ree = torch.mean(torch.abs(z - s) / s_safe)
        
    return ree

def compute_re_vectorized(theta, C, n_nodes):
    """
    Compute Reconstruction Error (RE) efficiently.
    RE = log(||L - P^T C^T L C P||_F^2 / n)
    """
    with torch.no_grad():
        # Compute pseudoinverse of C
        P = torch.linalg.pinv(C)
        
        # Compute lifted Laplacian: P^T @ C^T @ L @ C @ P
        L_coarse = C.T @ theta @ C
        L_lift = P.T @ L_coarse @ P
        
        # Reconstruction error
        diff = theta - L_lift
        re = torch.log(torch.norm(diff, 'fro').pow(2) / n_nodes + 1e-10)
        
    return re, L_lift

def compute_he_vectorized(L_lift, theta, X):
    """
    Compute Hyperbolic Error (HE) efficiently using vectorized operations.
    HE = acosh(1 + (||( L_lift - L) @ X||_F^2 * ||X||_F^2) / (2 * tr(X^T L_lift X) * tr(X^T L X)))
    """
    with torch.no_grad():
        # Ensure dense tensors
        L_lift_d = L_lift.to_dense() if L_lift.is_sparse else L_lift
        theta_d = theta.to_dense() if theta.is_sparse else theta
        X_d = X.to_dense() if X.is_sparse else X
        
        # Compute terms efficiently
        diff_X = (L_lift_d - theta_d) @ X_d
        norm_diff_sq = torch.norm(diff_X, 'fro').pow(2)
        norm_X_sq = torch.norm(X_d, 'fro').pow(2)
        numerator = norm_diff_sq * norm_X_sq
        
        # Traces using einsum for efficiency
        trace1 = torch.einsum('ij,jk,ki->', X_d.T, L_lift_d, X_d)
        trace2 = torch.einsum('ij,jk,ki->', X_d.T, theta_d, X_d)
        denominator = 2 * trace1 * trace2
        
        if denominator.abs() < 1e-9:
            return torch.tensor(0.0, device=theta.device)
        
        he = torch.acosh(1 + numerator / denominator)
        
    return he

def compute_all_metrics(theta, C, X, n_nodes, top_k=100):
    """
    Compute all metrics (REE, RE, HE) in one pass for efficiency.
    Returns: (ree, re, he)
    """
    ree = compute_ree_vectorized(theta, C, top_k=min(top_k, C.shape[1]))
    re, L_lift = compute_re_vectorized(theta, C, n_nodes)
    he = compute_he_vectorized(L_lift, theta, X)
    return ree, re, he

# --- Result Storage Initialization ---
accuracy_DGC1 = []
accuracy_DGC2 = []
accuracy_FGC = []
accuracy_without_coarsening = []
min_accuracy_DGC1_list = []
min_accuracy_DGC2_list = []
min_accuracy_FGC_list = []
REE_DGC1_list = []
REE_DGC2_list = []
REE_FGC_list = []
RE_DGC1_list = []
RE_DGC2_list = []
RE_FGC_list = []
hyp_DGC1_list = []
hyp_DGC2_list = []
hyp_FGC_list = []
time_DGC1_list = []
time_DGC2_list = []
time_FGC_list = []
peak_memory_DGC1_list = []
peak_memory_DGC2_list = []
peak_memory_FGC_list = []
diff_C_DGC1 = []
diff_C_DGC2 = []
diff2_C_DGC1 = []
diff2_C_DGC2 = []
delta_CS=[]
diff_adj=[]
modularity_FGC_list = []
modularity_DGC1_list = []
modularity_DGC2_list = []
conductance_FGC_list = []
conductance_DGC1_list = []
conductance_DGC2_list = []
f1_score_FGC=[]
f1_score_DGC1=[]
f1_score_DGC2=[]
nmi_FGC=[]
nmi_DGC1=[]
nmi_DGC2=[]

class MainProcess:
    def __init__(self):
        self.coarsening_ratio = 0.25 # Coarsening ratio
        self.k_coarsened = None # Will be set based on the initial graph size
        
        # Initialize the AS-733 loader
        self.dataset = AS733DynamicGraph() 
        self.max_iter = self.dataset.num_tasks # Should be 6
        self.num_classes = self.dataset.get_num_classes() # Dynamic classes
        print(f"Detected {self.num_classes} classes/communities.")
        
        self.hyperparameter_tuner = HyperparameterTuner(self.max_iter - 1)
        self.A_old = None
        self.size_old = None
        self.C_old = None
        self.X_old = None
        self.C_old2 = None
        
        # New: Track Node IDs to handle deletion/alignment
        self.ids_old = None

        # --- New storage for detailed logging ---
        self.graph_stats_list = []
        self.best_params_fgc_list = []
        self.best_params_dgc1_list = []
        self.best_params_dgc2_list = []


    def run(self):
        for i in range(self.max_iter):  
            
            # --- Load data for the current time step from AS-733 dataset ---
            edge_index, x, y, node_idxs = self.dataset.get_graph_for_timestep(i)
            
            if edge_index is None:
                print("All AS-733 time steps have been processed.")
                break

            print(f"\n{'='*20} Iteration {i} {'='*20}")
            stats = {'nodes': x.shape[0], 'edges': edge_index.shape[1]}
            self.graph_stats_list.append(stats)
            print(f"Graph stats: Nodes={stats['nodes']}, Edges={stats['edges']}, Features={x.shape[1]}")

            if i == 0:
                # --- First iteration: Baseline FGC ---
                print("\n--- Running Initial FGC Coarsening (T=0) ---")
                
                if x.shape[0] == 0:
                    print("Initial graph is empty. Skipping T=0.")
                    self.A_old = torch.empty((0,0))
                    self.size_old = 0
                    self.C_old = None
                    self.C_old2 = None
                    self.best_params_fgc_list.append({}) 
                    print("\n" + "="*50)
                    self.ids_old = torch.empty(0) 
                    continue

                self.A_old = to_dense_adj(edge_index)[0]
                labels = y.cpu().numpy()
                X = x.to_dense()
                
                # Store IDs for next iteration alignment
                self.ids_old = node_idxs 
                
                X, self.A_old, labels = preprocess_data(X, self.A_old, labels)
                theta = get_laplacian(self.A_old)

                p = X.shape[0]
                n = X.shape[1]
                
                self.size_old = p
                
                self.k_coarsened = int(p * self.coarsening_ratio) 
                print(f"Coarsening ratio: {self.coarsening_ratio}, Original nodes: {p}, Fixed coarsened nodes: {self.k_coarsened}")
                
                best_params, best_accuracy, self.C_old = self.hyperparameter_tuner.tune_hyperparameters(self.k_coarsened, n, p, theta, X, labels, self.num_classes, self.A_old, edge_index)
                self.best_params_fgc_list.append(best_params)
                self.C_old2 = self.C_old
                self.C_old.requires_grad = False
                self.C_old2.requires_grad = False
                
                print("\nInitial FGC tuning complete.")
                print("Best Parameters:", best_params)
                print(f"Best Accuracy for base FGC is: {best_accuracy:.4f}")
                print("\n" + "="*50)


            else:
                # --- Incremental iterations: FGC vs DIGC vs ACNR ---
                print(f"\n--- Running Incremental Update (T={i}) ---")

                if x.shape[0] == 0:
                    print(f"Graph at T={i} is empty. Skipping.")
                    self.A_old = torch.empty((0,0))
                    self.size_old = 0
                    print("\n" + "="*50)
                    self.ids_old = torch.empty(0)
                    continue

                # 1. Convert to dense for processing
                A_current_dense = to_dense_adj(edge_index)[0]
                X_current_dense = x.to_dense()
                
                # 2. ALIGNMENT & DELETION HANDLING
                print("Aligning graph nodes (handling additions/deletions)...")
                (
                    C_old_filtered, A_old_filtered, 
                    X_new, y_aligned, A_new, 
                    num_common, num_added,
                    ids_new_ordered
                ) = align_graph_components(
                    self.ids_old, node_idxs, 
                    self.C_old, self.A_old, 
                    X_current_dense, y, A_current_dense
                )
                
                # Align C_old2 as well (for DGC2)
                if self.C_old2 is not None:
                     C_old2_filtered, _, _, _, _, _, _, _ = align_graph_components(
                        self.ids_old, node_idxs, 
                        self.C_old2, None, 
                        X_current_dense, y, A_current_dense
                    )
                else:
                    C_old2_filtered = None

                # Update local references to use the ALIGNED data
                labels = y_aligned.cpu().numpy()
                
                # Preprocessing (normalization etc) on the ALIGNED data
                X_new, A_new, labels = preprocess_data(X_new, A_new, labels)
                theta_new = get_laplacian(A_new)

                size_new = X_new.shape[0]
                
                # CRITICAL: size_old is now the number of COMMON nodes
                self.size_old = num_common 
                new_nodes = num_added

                n = X_new.shape[1]
                p = X_new.shape[0]
                
                # 3. Super-node Cleanup (Dimension Reduction)
                C_old_filtered, k_dgc1 = prune_supernodes(C_old_filtered, self.k_coarsened)
                C_old2_filtered, k_dgc2 = prune_supernodes(C_old2_filtered, self.k_coarsened)
                
                print(f"Effective k for DGC1: {k_dgc1}, DGC2: {k_dgc2}")

                C_new = None
                C_new2 = None

                print(f'Preserved nodes: {self.size_old}, Added nodes: {new_nodes}, Total new size: {size_new}')
                
                if new_nodes <= 0:
                    print("Warning: No NEW nodes detected (only deletions or unchanged). Skipping incremental update.")
                    self.best_params_dgc1_list.append({})
                    self.best_params_dgc2_list.append({})
                    C_new = C_old_filtered
                    C_new2 = C_old2_filtered
                else:
                    # --- Run DIGC (DGC-1) ---
                    print("\n--- Tuning DIGC (DGC-1) ---")
                    best_params_dgc1, best_accuracy1,min_accuracy_DGC, C_new, time_DGC, peak_memory_DGC = self.hyperparameter_tuner.tune_hyperparameters_DGC(
                        (self.max_iter-2), (i-1), 
                        A_new, A_old_filtered, C_old_filtered, 
                        size_new, self.size_old, new_nodes, 
                        theta_new, labels, self.num_classes, k_dgc1, X_new, edge_index
                    )
                    self.best_params_dgc1_list.append(best_params_dgc1)
                    if C_new is not None:
                        C_new.requires_grad = False
                    accuracy_DGC1.append(best_accuracy1)
                    time_DGC1_list.append(time_DGC)
                    peak_memory_DGC1_list.append(peak_memory_DGC)
                    min_accuracy_DGC1_list.append(min_accuracy_DGC)
                    print(f"Best Accuracy for DIGC at T={i}: {best_accuracy1:.4f}")

                    # --- Run ACNR (DGC-2) ---
                    print("\n--- Tuning ACNR (DGC-2) ---")
                    
                    if C_old2_filtered is not None and A_old_filtered is not None:
                         dev = C_old2_filtered.device
                         A_old_f_dev = A_old_filtered.to(dev)
                         A_c_surv = C_old2_filtered.T @ A_old_f_dev @ C_old2_filtered
                    else:
                        A_c_surv = None

                    best_params_dgc2, best_accuracy2,min_accuracy_DGC, C_new2, time_DGC2, peak_memory_DGC2, cs, adj= self.hyperparameter_tuner.tune_hyperparameters_DGC2(
                        (self.max_iter-2), (i-1), 
                        A_new, A_old_filtered, C_old2_filtered, 
                        size_new, self.size_old, new_nodes, 
                        theta_new, labels, self.num_classes, k_dgc2, X_new, edge_index,
                        A_c_surv=A_c_surv 
                    )
                    self.best_params_dgc2_list.append(best_params_dgc2)
                    if C_new2 is not None:
                         C_new2.requires_grad = False
                    accuracy_DGC2.append(best_accuracy2)
                    time_DGC2_list.append(time_DGC2)
                    peak_memory_DGC2_list.append(peak_memory_DGC2)
                    min_accuracy_DGC2_list.append(min_accuracy_DGC)
                    delta_CS.append(cs)
                    diff_adj.append(adj)
                    print(f"Best Accuracy for ACNR at T={i}: {best_accuracy2:.4f}")
                    
                    if C_new is not None:
                        self.C_old = C_new
                        self.C_old.requires_grad = False
                    if C_new2 is not None:
                        self.C_old2 = C_new2
                        self.C_old2.requires_grad = False


                # --- Run FGC (Baseline from-scratch) ---
                print("\n--- Tuning FGC (from scratch) ---")
                best_params_fgc, best_accuracy_fgc, min_accuracy_FGC, C_FGC, time_FGC, peak_memory_FGC = self.hyperparameter_tuner.tune_hyperparameters_FGC((self.max_iter-1), (i-1), self.k_coarsened, n, p, theta_new, X_new, labels, self.num_classes, A_new, edge_index)
                self.best_params_fgc_list.append(best_params_fgc)
                accuracy_FGC.append(best_accuracy_fgc)
                time_FGC_list.append(time_FGC)
                peak_memory_FGC_list.append(peak_memory_FGC)
                min_accuracy_FGC_list.append(min_accuracy_FGC)
                print(f"Best Accuracy for FGC at T={i}: {best_accuracy_fgc:.4f}")
                
                # --- METRICS CALCULATION (REE, RE, HE) ---
                if new_nodes > 0 and C_new is not None and C_new2 is not None and C_FGC is not None:
                    print("\n--- Calculating Evaluation Metrics (REE, RE, HE) ---")
                    
                    # Determine top_k based on coarsened size (use min of 100 or k)
                    top_k = min(100, self.k_coarsened)
                    
                    # DGC-1 Metrics
                    try:
                        ree_dgc1, re_dgc1, he_dgc1 = compute_all_metrics(theta_new, C_new, X_new, p, top_k=top_k)
                        REE_DGC1_list.append(ree_dgc1)
                        RE_DGC1_list.append(re_dgc1)
                        hyp_DGC1_list.append(he_dgc1)
                        print(f"  DGC-1: REE={ree_dgc1.item():.6f}, RE={re_dgc1.item():.4f}, HE={he_dgc1.item():.6f}")
                    except Exception as e:
                        print(f"  DGC-1 metrics computation failed: {e}")
                    
                    # DGC-2 Metrics
                    try:
                        ree_dgc2, re_dgc2, he_dgc2 = compute_all_metrics(theta_new, C_new2, X_new, p, top_k=top_k)
                        REE_DGC2_list.append(ree_dgc2)
                        RE_DGC2_list.append(re_dgc2)
                        hyp_DGC2_list.append(he_dgc2)
                        print(f"  DGC-2: REE={ree_dgc2.item():.6f}, RE={re_dgc2.item():.4f}, HE={he_dgc2.item():.6f}")
                    except Exception as e:
                        print(f"  DGC-2 metrics computation failed: {e}")
                    
                    # FGC Metrics
                    try:
                        ree_fgc, re_fgc, he_fgc = compute_all_metrics(theta_new, C_FGC, X_new, p, top_k=top_k)
                        REE_FGC_list.append(ree_fgc)
                        RE_FGC_list.append(re_fgc)
                        hyp_FGC_list.append(he_fgc)
                        print(f"  FGC:   REE={ree_fgc.item():.6f}, RE={re_fgc.item():.4f}, HE={he_fgc.item():.6f}")
                    except Exception as e:
                        print(f"  FGC metrics computation failed: {e}")
                
                # --- Update state for next iteration ---
                self.A_old = A_new
                self.size_old = size_new
                self.ids_old = ids_new_ordered 
                print("\n" + "="*50)


        # --- Final Results ---
        print("\n\n--- FINAL RESULTS ---")
        
        print("\n--- Graph Statistics per Timestep ---")
        for t, stats in enumerate(self.graph_stats_list):
            print(f"T={t}: Nodes={stats['nodes']}, Edges={stats['edges']}")

        print('\n--- Accuracy Metrics ---')
        print('Accuracy for DGC-1 (DIGC) over time: ', [f'{acc:.4f}' for acc in accuracy_DGC1])
        print('Accuracy for DGC-2 (ACNR) over time: ', [f'{acc:.4f}' for acc in accuracy_DGC2])
        print('Accuracy for FGC (Baseline) over time: ', [f'{acc:.4f}' for acc in accuracy_FGC])

        print('\n Minimum Accuracy for DGC-1 (DIGC) over time: ', [f'{acc:.4f}' for acc in min_accuracy_DGC1_list])
        print('Minimum Accuracy for DGC-2 (ACNR) over time: ', [f'{acc:.4f}' for acc in min_accuracy_DGC2_list])
        print('Minimum Accuracy for FGC (Baseline) over time: ', [f'{acc:.4f}' for acc in min_accuracy_FGC_list])

        
        print('\n--- Performance Metrics ---')
        print('Time taken for DGC-1 (s): ', [f'{t:.2f}' for t in time_DGC1_list])
        print('Time taken for DGC-2 (s): ', [f'{t:.2f}' for t in time_DGC2_list])
        print('Time taken for FGC (s): ', [f'{t:.2f}' for t in time_FGC_list])
        print('\nPeak Memory for DGC-1 (MB): ', [f'{m:.2f}' for m in peak_memory_DGC1_list])
        print('Peak Memory for DGC-2 (MB): ', [f'{m:.2f}' for m in peak_memory_DGC2_list])
        print('Peak Memory for FGC (MB): ', [f'{m:.2f}' for m in peak_memory_FGC_list])

        print('\n--- Best Hyperparameters per Timestep ---')
        print(f"T=0 FGC: {self.best_params_fgc_list[0]}")
        for t in range(len(self.best_params_dgc1_list)):
            print(f"T={t+1} FGC:   {self.best_params_fgc_list[t+1]}")
            print(f"T={t+1} DGC-1: {self.best_params_dgc1_list[t]}")
            print(f"T={t+1} DGC-2: {self.best_params_dgc2_list[t]}")

        print('\n--- Graph Structure & Spectral Metrics ---')
        print('REE for DGC-1: ', [f'{val.item():.6f}' for val in REE_DGC1_list] if REE_DGC1_list else 'N/A')
        print('REE for DGC-2: ', [f'{val.item():.6f}' for val in REE_DGC2_list] if REE_DGC2_list else 'N/A')
        print('REE for FGC:   ', [f'{val.item():.6f}' for val in REE_FGC_list] if REE_FGC_list else 'N/A')
        print('\nRE for DGC-1: ', [f'{val.item():.4f}' for val in RE_DGC1_list] if RE_DGC1_list else 'N/A')
        print('RE for DGC-2: ', [f'{val.item():.4f}' for val in RE_DGC2_list] if RE_DGC2_list else 'N/A')
        print('RE for FGC:   ', [f'{val.item():.4f}' for val in RE_FGC_list] if RE_FGC_list else 'N/A')
        print('\nHE for DGC-1: ', [f'{val.item():.6f}' for val in hyp_DGC1_list] if hyp_DGC1_list else 'N/A')
        print('HE for DGC-2: ', [f'{val.item():.6f}' for val in hyp_DGC2_list] if hyp_DGC2_list else 'N/A')
        print('HE for FGC:   ', [f'{val.item():.6f}' for val in hyp_FGC_list] if hyp_FGC_list else 'N/A')


if __name__ == "__main__":
    main_process = MainProcess()
    main_process.run()

