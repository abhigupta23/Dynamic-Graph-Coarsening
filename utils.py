
import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rv_continuous

# Automatically detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_scipy_to_tensor(coo):
    try:
        coo = coo.tocoo()
    except:
        pass
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices).to(device)  # Move indices to the chosen device
    v = torch.FloatTensor(values).to(device)  # Move values to the chosen device
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)  # Ensure sparse tensor is also on device

class CustomDistribution(rv_continuous):
    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size)

def align_graph_components(ids_old, ids_new, C_old, A_old, X_new, y_new, A_new_dense):
    """
    Aligns the new graph components with the old ones by identifying common nodes
    and handling deletions/additions.
    
    Returns aligned matrices such that common nodes are first, followed by new nodes.
    """
    # Ensure inputs are numpy arrays for set operations
    ids_old_np = ids_old.cpu().numpy() if torch.is_tensor(ids_old) else ids_old
    ids_new_np = ids_new.cpu().numpy() if torch.is_tensor(ids_new) else ids_new
    
    # 1. Identify Common and New Nodes (Sorted by ID for deterministic behavior)
    common_ids = np.intersect1d(ids_old_np, ids_new_np)
    new_ids = np.setdiff1d(ids_new_np, ids_old_np)
    
    # 2. Create index mappings
    # Map global IDs to current local indices in the respective arrays
    old_id_to_idx = {uid: i for i, uid in enumerate(ids_old_np)}
    new_id_to_idx = {uid: i for i, uid in enumerate(ids_new_np)}
    
    # Indices in OLD structures to keep (rows of C_old / rows&cols of A_old)
    idx_old_keep = [old_id_to_idx[uid] for uid in common_ids]
    
    # Indices in NEW structures to reorder (Common first, then New)
    idx_new_common = [new_id_to_idx[uid] for uid in common_ids]
    idx_new_added = [new_id_to_idx[uid] for uid in new_ids]
    idx_new_perm = idx_new_common + idx_new_added
    
    # 3. Filter/Permute Old Structures (Handle Deletions)
    C_old_updated = None
    if C_old is not None:
        if isinstance(C_old, torch.Tensor):
            C_old_updated = C_old[idx_old_keep]
        else:
            C_old_updated = C_old # Fallback if not tensor (e.g. during first iter)
        
    A_old_updated = None
    if A_old is not None:
        if isinstance(A_old, torch.Tensor):
            A_old_updated = A_old[idx_old_keep][:, idx_old_keep]
        
    # 4. Permute New Structures (Align Common Nodes to Start)
    # Note: X_new, y_new, A_new_dense should be tensors
    perm_tensor = torch.tensor(idx_new_perm, device=X_new.device, dtype=torch.long)
    
    X_new_updated = X_new[perm_tensor]
    y_new_updated = y_new[perm_tensor]
    
    # Permute A_new_dense rows and cols
    A_new_updated = A_new_dense[perm_tensor][:, perm_tensor]
    
    # 5. Determine new 'effective' old size (number of preserved nodes)
    size_preserved = len(common_ids)
    size_added = len(new_ids)
    
    # Return the permuted IDs for next iteration's tracking
    if torch.is_tensor(ids_new):
         ids_new_ordered = ids_new[perm_tensor]
    else:
         ids_new_ordered = ids_new[np.array(idx_new_perm)]
    
    return (
        C_old_updated, A_old_updated, 
        X_new_updated, y_new_updated, A_new_updated, 
        size_preserved, size_added,
        ids_new_ordered
    )

def prune_supernodes(C, k_coarsened):
    """
    Removes empty supernodes (zero columns) from C and returns the updated C and new k.
    """
    if C is None:
        return C, k_coarsened
        
    # Check for columns that sum to 0 (empty supernodes)
    col_sums = C.sum(dim=0)
    nonzero_col_mask = col_sums > 1e-10
    
    if not nonzero_col_mask.all():
        # Filter out zero columns
        C_pruned = C[:, nonzero_col_mask]
        new_k = C_pruned.shape[1]
        print(f"Pruned {k_coarsened - new_k} empty supernodes. New k: {new_k}")
        return C_pruned, new_k
    
    return C, k_coarsened
