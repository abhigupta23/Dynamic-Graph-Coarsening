from data_loading import preprocess_data, get_laplacian
from utils import CustomDistribution, align_graph_components, prune_supernodes
from training import experiment, get_accuracy, get_accuracy_without_coarsening
from hyperparameter_tuning import HyperparameterTuner
from torch_geometric.utils import to_dense_adj
import numpy as np
import torch
from models import Net
from acm_data_loader import ACMDynamicGraph # <-- IMPORT THE UPDATED ACM DATA LOADER
from sklearn.metrics import normalized_mutual_info_score # <-- IMPORT FOR NMI
import os
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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
        # Initialize the ACM loader with the new temporal logic
        self.dataset = ACMDynamicGraph() 
        self.max_iter = self.dataset.num_tasks # Will be 5
        self.hyperparameter_tuner = HyperparameterTuner(self.max_iter - 1)
        self.A_old = None
        self.size_old = None
        self.C_old = None
        self.X_old = None
        self.C_old2 = None
        self.num_classes = 4 
        
        # New: Track Node IDs to handle deletion/alignment
        self.ids_old = None

        # --- New storage for detailed logging ---
        self.graph_stats_list = []
        self.best_params_fgc_list = []
        self.best_params_dgc1_list = []
        self.best_params_dgc2_list = []


    def run(self):
        for i in range(self.max_iter):  
            
            # --- Load data for the current time step from ACM dataset ---
            # UPDATED: Now unpacks 4 values including node_idxs
            edge_index, x, y, node_idxs = self.dataset.get_graph_for_timestep(i)
            
            if edge_index is None:
                print("All ACM time steps have been processed.")
                break

            print(f"\n{'='*20} Iteration {i} {'='*20}")
            stats = {'nodes': x.shape[0], 'edges': edge_index.shape[1]}
            self.graph_stats_list.append(stats)
            print(f"Graph stats: Nodes={stats['nodes']}, Edges={stats['edges']}, Features={x.shape[1]}")

            if i == 0:
                # --- First iteration: Baseline FGC on 1990-1993 data ---
                print("\n--- Running Initial FGC Coarsening (T=0) ---")
                
                if x.shape[0] == 0:
                    print("Initial graph is empty. Skipping T=0.")
                    self.A_old = torch.empty((0,0))
                    self.size_old = 0
                    self.C_old = None
                    self.C_old2 = None
                    self.best_params_fgc_list.append({}) # Append empty dict for consistency
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
                # We align the current graph to put "preserved" nodes first.
                # We also filter C_old and A_old to remove deleted nodes.
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
                # If deletion resulted in empty supernodes, prune them.
                # This modifies k for the respective method.
                
                # DGC1 Cleanup
                C_old_filtered, k_dgc1 = prune_supernodes(C_old_filtered, self.k_coarsened)
                # DGC2 Cleanup
                C_old2_filtered, k_dgc2 = prune_supernodes(C_old2_filtered, self.k_coarsened)
                
                # Note: We use the pruned k for optimization
                print(f"Effective k for DGC1: {k_dgc1}, DGC2: {k_dgc2}")

                C_new = None
                C_new2 = None

                print(f'Preserved nodes: {self.size_old}, Added nodes: {new_nodes}, Total new size: {size_new}')
                
                if new_nodes <= 0:
                    print("Warning: No NEW nodes detected (only deletions or unchanged). Skipping incremental update.")
                    # If we only have deletions, we might still want to return the filtered C_old as the new C?
                    self.best_params_dgc1_list.append({})
                    self.best_params_dgc2_list.append({})
                    
                    # If only deletions, C_old_filtered is effectively the new C (though reduced size)
                    C_new = C_old_filtered
                    C_new2 = C_old2_filtered
                else:
                    # --- Run DIGC (DGC-1) ---
                    print("\n--- Tuning DIGC (DGC-1) ---")
                    # Pass filtered/aligned matrices
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
                    
                    # ACNR PRE-STEP: Compute Reference Coarsened Adjacency on Survivors
                    # A_c_old_surv = (C_old_surv)^T @ A_old_surv @ C_old_surv
                    if C_old2_filtered is not None and A_old_filtered is not None:
                        # Ensure on same device
                         dev = C_old2_filtered.device
                         A_old_f_dev = A_old_filtered.to(dev)
                         # Calculate reference
                         A_c_surv = C_old2_filtered.T @ A_old_f_dev @ C_old2_filtered
                    else:
                        A_c_surv = None

                    best_params_dgc2, best_accuracy2,min_accuracy_DGC, C_new2, time_DGC2, peak_memory_DGC2, cs, adj= self.hyperparameter_tuner.tune_hyperparameters_DGC2(
                        (self.max_iter-2), (i-1), 
                        A_new, A_old_filtered, C_old2_filtered, 
                        size_new, self.size_old, new_nodes, 
                        theta_new, labels, self.num_classes, k_dgc2, X_new, edge_index,
                        A_c_surv=A_c_surv # PASS THE RECALIBRATED REFERENCE
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
                    
                    # Plot convergence (one set of hyperparameters = best run)
                    if plt is not None:
                        try:
                            os.makedirs('convergence_plots', exist_ok=True)
                            # DGC-1
                            loss_hist_dgc1 = self.hyperparameter_tuner.loss_history_dgc[i-1] if (i-1) < len(self.hyperparameter_tuner.loss_history_dgc) else None
                            if loss_hist_dgc1 is not None:
                                import numpy as _np
                                _logy = _np.log10(_np.array(loss_hist_dgc1) + 1e-12)
                                plt.figure()
                                plt.plot(range(1, len(_logy)+1), _logy, label='DGC-1 (SGD)')
                                plt.xlabel('Epoch')
                                plt.ylabel('log10(Objective)')
                                plt.title(f'Convergence at T={i} - DGC-1')
                                plt.legend()
                                plt.tight_layout()
                                plt.savefig(f'convergence_plots/T{i}_DGC1.png')
                                plt.close()
                                # Normalized + smoothed plot
                                try:
                                    import numpy as _np
                                    base = loss_hist_dgc1[0]
                                    norm = [(v - base)/ (abs(base) + 1e-8) for v in loss_hist_dgc1]
                                    if len(norm) >= 5:
                                        w = 5
                                        kernel = _np.ones(w)/w
                                        smooth = _np.convolve(_np.array(norm), kernel, mode='valid')
                                    else:
                                        smooth = norm
                                    plt.figure()
                                    plt.plot(range(1, len(norm)+1), norm, alpha=0.3, label='Normalized')
                                    plt.plot(range(1, len(smooth)+1), smooth, color='red', label='Normalized (MA-5)')
                                    plt.xlabel('Epoch')
                                    plt.ylabel('Normalized Objective')
                                    plt.title(f'Normalized Convergence at T={i} - DGC-1')
                                    plt.legend()
                                    plt.tight_layout()
                                    plt.savefig(f'convergence_plots/T{i}_DGC1_norm.png')
                                    plt.close()
                                except Exception as _e:
                                    pass
                            # DGC-2
                            loss_hist_dgc2 = self.hyperparameter_tuner.loss_history_dgc2[i-1] if (i-1) < len(self.hyperparameter_tuner.loss_history_dgc2) else None
                            if loss_hist_dgc2 is not None:
                                import numpy as _np
                                _logy = _np.log10(_np.array(loss_hist_dgc2) + 1e-12)
                                plt.figure()
                                plt.plot(range(1, len(_logy)+1), _logy, color='orange', label='DGC-2 (SGD)')
                                plt.xlabel('Epoch')
                                plt.ylabel('log10(Objective)')
                                plt.title(f'Convergence at T={i} - DGC-2')
                                plt.legend()
                                plt.tight_layout()
                                plt.savefig(f'convergence_plots/T{i}_DGC2.png')
                                plt.close()
                                # Normalized + smoothed plot
                                try:
                                    import numpy as _np
                                    base = loss_hist_dgc2[0]
                                    norm = [(v - base)/ (abs(base) + 1e-8) for v in loss_hist_dgc2]
                                    if len(norm) >= 5:
                                        w = 5
                                        kernel = _np.ones(w)/w
                                        smooth = _np.convolve(_np.array(norm), kernel, mode='valid')
                                    else:
                                        smooth = norm
                                    plt.figure()
                                    plt.plot(range(1, len(norm)+1), norm, alpha=0.3, label='Normalized')
                                    plt.plot(range(1, len(smooth)+1), smooth, color='red', label='Normalized (MA-5)')
                                    plt.xlabel('Epoch')
                                    plt.ylabel('Normalized Objective')
                                    plt.title(f'Normalized Convergence at T={i} - DGC-2')
                                    plt.legend()
                                    plt.tight_layout()
                                    plt.savefig(f'convergence_plots/T{i}_DGC2_norm.png')
                                    plt.close()
                                except Exception as _e:
                                    pass
                        except Exception as e:
                            print(f"Convergence plotting failed at T={i}: {e}")
                    
                    if C_new is not None:
                        self.C_old = C_new
                        self.C_old.requires_grad = False
                    if C_new2 is not None:
                        self.C_old2 = C_new2
                        self.C_old2.requires_grad = False


                # --- Run FGC (Baseline from-scratch) ---
                print("\n--- Tuning FGC (from scratch) ---")
                # FGC doesn't care about alignment history, it just runs on current X_new/A_new
                best_params_fgc, best_accuracy_fgc, min_accuracy_FGC, C_FGC, time_FGC, peak_memory_FGC = self.hyperparameter_tuner.tune_hyperparameters_FGC((self.max_iter-1), (i-1), self.k_coarsened, n, p, theta_new, X_new, labels, self.num_classes, A_new, edge_index)
                self.best_params_fgc_list.append(best_params_fgc)
                accuracy_FGC.append(best_accuracy_fgc)
                time_FGC_list.append(time_FGC)
                peak_memory_FGC_list.append(peak_memory_FGC)
                min_accuracy_FGC_list.append(min_accuracy_FGC)
                print(f"Best Accuracy for FGC at T={i}: {best_accuracy_fgc:.4f}")
                
                # --- METRICS CALCULATION ---
                if new_nodes > 0 and C_new is not None and C_new2 is not None:
                    print("\n--- Calculating Evaluation Metrics ---")
                    
                    # eigen_values,eigenvectors=torch.linalg.eig(theta_new)
                    # eigen_values_magnitude = torch.abs(eigen_values)
                    
                    # sorted_mag_s,indices=torch.sort(eigen_values_magnitude)
                    # s = eigen_values[indices]

                    # eigen_value,eigenvector=torch.linalg.eig(torch.transpose(C_new,0,1)@theta_new@C_new)
                    # sorted_indices = torch.argsort(torch.abs(eigen_value))
                    # z = eigen_value[sorted_indices]

                    # s_new=s[-100:]
                    # z_new=z[-100:]

                    # temp=0
                    # for j in range(len(s_new)):
                    #     temp=temp+(abs(z_new[j]-s_new[j])/s_new[j])
                    # eigenerror=temp/len(s_new)
                    # print(f'REE of DGC - 1 algorithm is: {eigenerror}')
                    # REE_DGC1_list.append(eigenerror)
                    
                    # eigen_value,eigenvector=torch.linalg.eig(torch.transpose(C_new2,0,1)@theta_new@C_new2)
                    # sorted_indices = torch.argsort(torch.abs(eigen_value))
                    # z = eigen_value[sorted_indices]

                    # z_new=z[-100:]

                    # temp=0
                    # for j in range(len(s_new)):
                    #     temp=temp+(abs(z_new[j]-s_new[j])/s_new[j])
                    # eigenerror=temp/len(s_new)
                    # print(f'REE of DGC - 2 algorithm is: {eigenerror}')
                    # REE_DGC2_list.append(eigenerror)
                    
                    # t_nodes = p 
                    # P=torch.linalg.pinv(C_new)
                    # L_lift=torch.transpose(P,0,1)@torch.transpose(C_new,0,1)@theta_new@C_new@P
                    # LL=(theta_new-L_lift)
                    # RE_DGC = torch.log(pow(torch.linalg.norm(LL),2)/t_nodes)
                    # print('Reconstruction error of DGC - 1', RE_DGC)
                    # RE_DGC1_list.append(RE_DGC)
                    
                    # P=torch.linalg.pinv(C_new2)
                    # L_lift2=torch.transpose(P,0,1)@torch.transpose(C_new2,0,1)@theta_new@C_new2@P
                    # LL2=(theta_new-L_lift2)
                    # RE_DGC = torch.log(pow(torch.linalg.norm(LL2),2)/t_nodes)
                    # print('Reconstruction error of DGC - 2', RE_DGC)
                    # RE_DGC2_list.append(RE_DGC)
                    
                    # def HE(u,v, X_matrix):
                    #     u_dense = u.to_dense() if u.is_sparse else u
                    #     v_dense = v.to_dense() if v.is_sparse else v
                    #     X_dense = X_matrix.to_dense() if X_matrix.is_sparse else X_matrix
                        
                    #     term1 = (u_dense - v_dense) @ X_dense
                    #     norm1_sq = torch.norm(term1, 'fro')**2
                    #     norm_X_sq = torch.norm(X_dense, 'fro')**2
                    #     numerator = norm1_sq * norm_X_sq
                        
                    #     trace1 = torch.trace(X_dense.T @ u_dense @ X_dense)
                    #     trace2 = torch.trace(X_dense.T @ v_dense @ X_dense)
                    #     denominator = 2 * trace1 * trace2
                        
                    #     if denominator.abs() < 1e-9: return torch.tensor(0.0)
                        
                    #     return torch.acosh(1 + numerator / denominator)

                    # hyp_DGC1 = HE(L_lift,theta_new, X_new)
                    # print('Hyperbolic error for DGC - 1:', hyp_DGC1)
                    # hyp_DGC1_list.append(hyp_DGC1)

                    # hyp_DGC2 = HE(L_lift2,theta_new, X_new)
                    # print('Hyperbolic error for DGC - 2:', hyp_DGC2)
                    # hyp_DGC2_list.append(hyp_DGC2)
                    
                    # eigen_value,eigenvector=torch.linalg.eig(torch.transpose(C_FGC,0,1)@theta_new@C_FGC)
                    # sorted_indices = torch.argsort(torch.abs(eigen_value))
                    # z = eigen_value[sorted_indices]
                    # z_new=z[-100:]

                    # temp=0
                    # for j in range(len(s_new)):
                    #     temp=temp+(abs(z_new[j]-s_new[j])/s_new[j])
                    # eigenerror=temp/len(s_new)
                    # print(f'REE of FGC algorithm is: {eigenerror}')
                    # REE_FGC_list.append(eigenerror)
                    
                    # P_fgc=torch.linalg.pinv(C_FGC)
                    # L_lift_fgc=torch.transpose(P_fgc,0,1)@torch.transpose(C_FGC,0,1)@theta_new@C_FGC@P_fgc
                    # LL_fgc=(theta_new-L_lift_fgc)
                    # RE_FGC = torch.log(pow(torch.linalg.norm(LL_fgc),2)/t_nodes)
                    # print('Reconstruction error of FGC', RE_FGC)
                    # RE_FGC_list.append(RE_FGC)
                    
                    # hyp_FGC = HE(L_lift_fgc,theta_new, X_new)
                    # print('Hyperbolic error for FGC:', hyp_FGC)
                    # hyp_FGC_list.append(hyp_FGC)

                    # def modularity_torch(adjacency, C):
                    #     clusters = torch.argmax(C, dim=1)
                    #     degrees = adjacency.sum(dim=0)
                    #     n_edges = degrees.sum()
                    #     if n_edges == 0: return 0.0
                    #     result = 0.0
                    #     unique_clusters = torch.unique(clusters)
                    #     for cluster_id in unique_clusters:
                    #         cluster_mask = (clusters == cluster_id)
                    #         adj_submatrix = adjacency[cluster_mask, :][:, cluster_mask]
                    #         degrees_submatrix = degrees[cluster_mask]
                    #         result += adj_submatrix.sum() - (degrees_submatrix.sum() ** 2) / n_edges
                    #     return result / n_edges

                    # def conductance_torch(adjacency, C):
                    #     clusters = torch.argmax(C, dim=1)
                    #     inter, intra = 0.0, 0.0
                    #     unique_clusters = torch.unique(clusters)
                    #     for cluster_id in unique_clusters:
                    #         cluster_mask = (clusters == cluster_id)
                    #         adj_submatrix = adjacency[cluster_mask, :]
                    #         intra += adj_submatrix[:, cluster_mask].sum().item()
                    #         inter += adj_submatrix[:, ~cluster_mask].sum().item()
                    #     return intra / (inter + intra) if (inter + intra) > 0 else 0.0
                    
                    # def _pairwise_confusion(y_true, y_pred):
                    #     n = y_true.size(0)
                    #     contingency = _torch_contingency_matrix(y_true, y_pred)
                    #     same_class_true = contingency.max(dim=1).values
                    #     same_class_pred = contingency.max(dim=0).values
                    #     diff_class_true = contingency.sum(dim=1) - same_class_true
                    #     diff_class_pred = contingency.sum(dim=0) - same_class_pred
                    #     total = contingency.sum()
                    #     true_positives = (same_class_true * (same_class_true - 1)).sum()
                    #     false_positives = (diff_class_true * same_class_true * 2).sum()
                    #     false_negatives = (diff_class_pred * same_class_pred * 2).sum()
                    #     true_negatives = total * (total - 1) - true_positives - false_positives - false_negatives
                    #     return true_positives, false_positives, false_negatives, true_negatives

                    # def _torch_contingency_matrix(y_true, y_pred):
                    #     max_true = y_true.max().item() + 1
                    #     max_pred = y_pred.max().item() + 1
                    #     contingency = torch.zeros((max_true, max_pred), dtype=torch.float32, device=y_true.device)
                    #     for t, p in zip(y_true, y_pred):
                    #         contingency[t, p] += 1
                    #     return contingency

                    # def pairwise_f1_score(y_true, y_pred):
                    #     true_positives, false_positives, false_negatives, _ = _pairwise_confusion(y_true, y_pred)
                    #     precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                    #     recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                    #     if precision + recall == 0: return torch.tensor(0.0, dtype=torch.float32)
                    #     return 2 * (precision * recall) / (precision + recall)

                    # def loading_matrix_to_labels(loading_matrix):
                    #     return torch.argmax(loading_matrix, dim=1)

                    # def compute_nmi_torch(y_true, y_pred):
                    #     y_true_list = y_true.cpu().tolist()
                    #     y_pred_list = y_pred.cpu().tolist()
                    #     return torch.tensor(normalized_mutual_info_score(y_true_list, y_pred_list))

                    # y_true = torch.from_numpy(labels).to(A_new.device)
                    # y_pred_fgc = loading_matrix_to_labels(C_FGC)
                    # y_pred_dgc1 = loading_matrix_to_labels(C_new)
                    # y_pred_dgc2 = loading_matrix_to_labels(C_new2)

                    # f1_score_FGC.append(pairwise_f1_score(y_true, y_pred_fgc))
                    # f1_score_DGC1.append(pairwise_f1_score(y_true, y_pred_dgc1))
                    # f1_score_DGC2.append(pairwise_f1_score(y_true, y_pred_dgc2))
                    # nmi_FGC.append(compute_nmi_torch(y_true, y_pred_fgc))
                    # nmi_DGC1.append(compute_nmi_torch(y_true, y_pred_dgc1))
                    # nmi_DGC2.append(compute_nmi_torch(y_true, y_pred_dgc2))
                    # modularity_FGC_list.append(modularity_torch(A_new, C_FGC))
                    # modularity_DGC1_list.append(modularity_torch(A_new, C_new))
                    # modularity_DGC2_list.append(modularity_torch(A_new, C_new2))
                    # conductance_FGC_list.append(conductance_torch(A_new, C_FGC))
                    # conductance_DGC1_list.append(conductance_torch(A_new, C_new))
                    # conductance_DGC2_list.append(conductance_torch(A_new, C_new2))
                    pass

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

        print('\n--- Graph Structure & Clustering Metrics ---')
        print('REE for DGC-1: ', [f'{val.item():.4f}' for val in REE_DGC1_list])
        print('REE for DGC-2: ', [f'{val.item():.4f}' for val in REE_DGC2_list])
        print('REE for FGC:   ', [f'{val.item():.4f}' for val in REE_FGC_list])
        print('\nRE for DGC-1: ', [f'{val.item():.4f}' for val in RE_DGC1_list])
        print('RE for DGC-2: ', [f'{val.item():.4f}' for val in RE_DGC2_list])
        print('RE for FGC:   ', [f'{val.item():.4f}' for val in RE_FGC_list])
        print('\nHE for DGC-1: ', [f'{val.item():.4f}' for val in hyp_DGC1_list])
        print('HE for DGC-2: ', [f'{val.item():.4f}' for val in hyp_DGC2_list])
        print('HE for FGC:   ', [f'{val.item():.4f}' for val in hyp_FGC_list])
        print('\nF1 Score for DGC-1: ', [f'{val.item():.4f}' for val in f1_score_DGC1])
        print('F1 Score for DGC-2: ', [f'{val.item():.4f}' for val in f1_score_DGC2])
        print('F1 Score for FGC:   ', [f'{val.item():.4f}' for val in f1_score_FGC])
        print('\nNMI for DGC-1: ', [f'{val.item():.4f}' for val in nmi_DGC1])
        print('NMI for DGC-2: ', [f'{val.item():.4f}' for val in nmi_DGC2])
        print('NMI for FGC:   ', [f'{val.item():.4f}' for val in nmi_FGC])
        print('\nModularity for DGC-1: ', [f'{val:.4f}' for val in modularity_DGC1_list])
        print('Modularity for DGC-2: ', [f'{val:.4f}' for val in modularity_DGC2_list])
        print('Modularity for FGC:   ', [f'{val:.4f}' for val in modularity_FGC_list])
        print('\nConductance for DGC-1: ', [f'{val:.4f}' for val in conductance_DGC1_list])
        print('Conductance for DGC-2: ', [f'{val:.4f}' for val in conductance_DGC2_list])
        print('Conductance for FGC:   ', [f'{val:.4f}' for val in conductance_FGC_list])


if __name__ == "__main__":
    main_process = MainProcess()
    main_process.run()
