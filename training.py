from __future__ import annotations
import torch
import torch.nn.functional as F
from tqdm import tqdm
from random import sample
from datetime import datetime
from scipy.sparse import csr_matrix
import numpy as np
from utils import convert_scipy_to_tensor
from scipy import sparse
from data_loading import get_block_matrices, get_feature_new_nodes, get_block_matrices2
from scipy import sparse
import tracemalloc
# import dgl
from sklearn.metrics import roc_auc_score


def experiment(lambda_param: float, beta_param: float, alpha_param: float, gamma_param: float, C: torch.Tensor, X_tilde: torch.Tensor, theta: torch.Tensor, X: torch.Tensor, k: int, p: int, thresh: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ones = torch.ones((k, k), dtype=C.dtype)
    # J = torch.ones((k, k), dtype=C.dtype) / k
    # zeros = torch.zeros((p, k), dtype=C.dtype)
    # eye = torch.eye(k, dtype=C.dtype)
    device = C.device
    ones = torch.ones((k, k), dtype=C.dtype, device=device)
    J = torch.ones((k, k), dtype=C.dtype, device=device) / k
    zeros = torch.zeros((p, k), dtype=C.dtype, device=device)
    eye = torch.eye(k, dtype=C.dtype, device=device)

    def update(X_tilde, C, i, L):
        thetaC = theta @ C
        # X_tilde_new=None
        CT = torch.transpose(C, 0, 1)
        X_tildeT = torch.transpose(X_tilde, 0, 1)
        CX_tilde = C @ X_tilde
        t1 = CT @ thetaC + J
        L = 1 / k
        check = 0
        try:
            term_bracket = torch.linalg.pinv(t1)
        except:
            check = 1
        if check == 0:
            thetacX_tilde = thetaC @ X_tilde
            t1 = -2 * gamma_param * (thetaC @ term_bracket)
            t2 = alpha_param * (CX_tilde.to_dense() - X.to_dense()) @ X_tildeT.to_dense()
            t3 = 2 * thetacX_tilde @ X_tildeT.to_dense()
            t4 = lambda_param * (C @ ones)
            t5 = 2 * beta_param * (thetaC @ CT @ thetaC)
            T2 = (t1 + t2 + t3 + t4 + t5) / L
            Cnew = (C.to_dense() - T2).maximum(zeros)
            t1 = CT @ thetaC * (2 / alpha_param)
            t2 = CT @ C
            check2 = 0
            try:
                t1 = torch.linalg.pinv(t1 + t2)
            except:
                check2 = 1
            if check2 == 0:
                t1 = t1 @ CT
                t1 = t1 @ X.to_dense()
                X_tilde_new = t1
                Cnew[Cnew < thresh] = thresh
                for i in range(len(Cnew)):
                    Cnew[i] = Cnew[i] / torch.linalg.norm(Cnew[i], 1)
                for i in range(len(X_tilde_new)):
                    X_tilde_new[i] = X_tilde_new[i] / torch.linalg.norm(X_tilde_new[i], 1)
                return X_tilde_new, Cnew, L
            else:
                return X_tilde, Cnew, L
        else:
            return X_tilde, C, L

    for i in tqdm(range(5)):
        X_tilde, C, L = update(X_tilde, C, i, None)
    # X_tilde_old = X_tilde
    return X_tilde, C

def experiment_deltaC(X_N: torch.Tensor, X_old_super: torch.Tensor, M2: torch.Tensor, M3: torch.Tensor, M4: torch.Tensor, alpha_param: float, beta_param: float, C_old: torch.Tensor, delta_C: torch.Tensor, k: int, new_nodes: int) -> torch.Tensor:
    device = X_N.device
    ones = torch.ones((k, k), dtype=C_old.dtype, device=device)
    zeros = torch.zeros((new_nodes, k), dtype=C_old.dtype, device=device)
    C_oldT = torch.transpose(C_old, 0, 1).to(device)
    eta0 = 0.01
    tol = 1e-6
    thresh = 1e-10
    M2T = torch.transpose(M2, 0, 1).to(device)
    M2T_Cold = M2T @ C_old
    M3_C = M3 @ C_old
    X_old_superT = torch.transpose(X_old_super, 0, 1).to(device)

    for i in range(20):
        delta_CT = torch.transpose(delta_C, 0, 1).to(device)
        M4_deltaC = M4 @ delta_C.to(device)
        t1_b1 = M2T @ C_old
        t2_b1 = M4 @ delta_C.to(device)
        t3_b2 = delta_CT @ M2T @ C_old
        t4_b2 = C_oldT @ M2 @ delta_C.to(device)
        t5_b2 = delta_CT @ M4 @ delta_C.to(device)
        t6_b3 = delta_C @ X_old_super.to(device)
        t7_b3 = X_N.to(device)

        b1 = t1_b1 + t2_b1
        b2 = t3_b2 + t4_b2 + t5_b2
        b3 = t6_b3 + t7_b3

        T1 = -2 * alpha_param * (b1 @ b2)
        T2 = beta_param * (b3 @ X_old_superT)
        differentiation = T1 + T2
        delta_C_dense = delta_C.to_dense() if delta_C.is_sparse else delta_C
        differentiation_dense = differentiation.to_dense() if differentiation.is_sparse else differentiation
        delta_C_dense = delta_C_dense - eta0 * differentiation_dense
        delta_C_dense = delta_C_dense.maximum(zeros)
        delta_C_dense[delta_C_dense < thresh] = thresh
        for j in range(len(delta_C_dense)):
            delta_C_dense[j] = delta_C_dense[j] / torch.linalg.norm(delta_C_dense[j], 1)

        grad_norm = torch.norm(differentiation_dense)
        if grad_norm < tol:
            print(f"Convergence reached after {i + 1} iterations.")
            break

        delta_C = delta_C_dense
    return delta_C


def DGC(alpha_param: float, beta_param: float, delta_C: torch.Tensor, A_new: torch.Tensor, A_old: torch.Tensor, C_old: torch.Tensor, size_new: int, size_old: int, new_nodes: int, X_new: torch.Tensor,X_old_super: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    M1, M2, M3, M4 = get_block_matrices(A_old, A_new, size_old, new_nodes)
    X_N = get_feature_new_nodes(X_new, new_nodes)

    # Configure SGD optimizer (stochastic gradient descent)
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    max_epochs = 200
    min_epochs = 20
    rel_tol = 1e-4
    patience = 5
    thresh = 1e-10

    delta_C = torch.nn.Parameter(delta_C)
    C_old.requires_grad = False
    optimizer = torch.optim.SGD([delta_C], lr=learning_rate, momentum=momentum, nesterov=True, weight_decay=weight_decay)

    loss_history = []

    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prev_loss = None
    stable_count = 0
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss = DGC_objective_function(delta_C, C_old, M2, M4, X_old_super, X_N, alpha_param, beta_param)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_C], max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            delta_C.data = delta_C.data.clamp(min=thresh)
            for j in range(delta_C.data.size(0)):
                norm_j = torch.linalg.norm(delta_C.data[j], 1)
                if norm_j > 0:
                    delta_C.data[j] = delta_C.data[j] / norm_j

        cur_loss = loss.detach().item()
        loss_history.append(cur_loss)
        if prev_loss is not None and epoch + 1 >= min_epochs:
            rel_change = abs(prev_loss - cur_loss) / (abs(prev_loss) + 1e-8)
            if rel_change < rel_tol:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= patience:
                break
        prev_loss = cur_loss

    current, peak = tracemalloc.get_traced_memory()
    if torch.cuda.is_available():
        peak_gpu = torch.cuda.max_memory_allocated(device="cuda") 
    else:
        peak_gpu = 0
    peak += peak_gpu
    tracemalloc.stop()
    C_new = torch.cat((C_old, delta_C.detach()), dim=0)
    C_new = C_new.detach()
    try:
        X_tilde_new = torch.linalg.pinv(C_new) @ X_new.to_dense()
        status = True
        return X_tilde_new, C_new, peak, status, loss_history
    except:
        status = False
        return 0, 0, 0, status, loss_history


def DGC_objective_function(Delta_C_t2, C_t1, M2, M4, X_sn, X_n, omega, zeta):
    """
    Compute the objective function value using PyTorch tensors.
    """
    # Compute \\( \\mathbf{A}_{c_{t_2}} - \\mathbf{A}_{c_{t_1}} \\)
    
    term_1 = Delta_C_t2.T @ M2.T @ C_t1
    term_2 = C_t1.T @ M2 @ Delta_C_t2
    term_3 = Delta_C_t2.T @ M4 @ Delta_C_t2
    # print(term_1.shape, term_2.shape, term_3.shape)
    A_diff = term_1 + term_2 + term_3

    # Compute \\( -\\omega \\| \\mathbf{A}_{c_{t_2}} - \\mathbf{A}_{c_{t_1}} \\|_F^2 \\)
    term_omega = omega * torch.norm(A_diff, p='fro')**2

    # Compute \\( \\| \\Delta \\mathbf{C}_{t_2} \\mathbf{1} - \\mathbf{1} \\|_F^2 \\)
    ones = torch.ones((Delta_C_t2.shape[1], 1), device=Delta_C_t2.device)
    term_delta_C = torch.norm(Delta_C_t2 @ ones - torch.ones((Delta_C_t2.shape[0], 1), device=Delta_C_t2.device), p='fro')**2

    # Compute \\( \\zeta \\| \\Delta \\mathbf{C}_{t_2} \\mathbf{X}_{\\text{sn}} - \\mathbf{X}_n \\|_F^2 \\)
    # print(Delta_C_t2.shape, X_sn.shape, X_n.shape)
    term_zeta = zeta * torch.norm(Delta_C_t2 @ X_sn - X_n, p='fro')**2

    # Combine terms to compute the objective
    objective_value = term_omega + term_delta_C + term_zeta

    return objective_value
    
def get_accuracy_without_coarsening(L: torch.Tensor, labels: np.ndarray, num_classes: int, X: torch.Tensor, adj: torch.Tensor, model_class, num_epochs: int = 100):
    # Set the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert labels to tensor and move to the appropriate device
    labels = torch.tensor(labels, device=device)

    # Prepare weighted adjacency matrix
    Wc = (-1 * L) * (1 - torch.eye(L.shape[0], device=device))
    Wc[Wc < 0.1] = 0
    Wc = Wc.cpu().detach().numpy()
    Wc = sparse.csr_matrix(Wc)
    Wc = Wc.tocoo()
    row = torch.from_numpy(Wc.row).to(device).long()
    col = torch.from_numpy(Wc.col).to(device).long()
    edge_index_coarsen2 = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(Wc.data).to(device)

    # Function for one-hot encoding
    def one_hot(x, class_count):
        return torch.eye(class_count, dtype=X.dtype, device=device)[x, :]

    Y = one_hot(labels, num_classes)
    
    # Randomly zero out 20% of the rows in Y for training/testing masks
    num_rows_to_zero = int(0.2 * Y.size(0))
    indices_to_zero = sample(range(0, Y.size(0)), num_rows_to_zero)
    train_mask = torch.ones(Y.size(0), dtype=torch.bool, device=device)
    train_mask[indices_to_zero] = False
    Y[~train_mask] = 0
    test_mask = ~train_mask

    # Initialize model and optimizer
    model = model_class(X.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(torch.Tensor(X.to_dense()).to(device), edge_index_coarsen2)
        loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        return loss

    # Training loop
    for epoch in range(num_epochs):
        loss = train()
        if epoch % 100 == 0:
            print(f'Epoch: {epoch:03d}, loss: {loss:.4f}')

    # Prepare the edge index for the adjacency matrix
    Wc = sparse.csr_matrix(adj.cpu().numpy())
    Wc = Wc.tocoo()
    row = torch.from_numpy(Wc.row).to(device).long()
    col = torch.from_numpy(Wc.col).to(device).long()
    edge_index_coarsen = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(Wc.data).to(device)

    # Prediction and accuracy calculation
    pred = model(torch.Tensor(X.to_dense()).to(device), edge_index_coarsen).argmax(dim=1)
    correct = (pred[test_mask] == labels[test_mask]).sum().item()
    acc = correct / num_rows_to_zero
    return acc

def get_accuracy(C_0: torch.Tensor, L: torch.Tensor, X_t_0: torch.Tensor, labels: np.ndarray, num_classes: int, k: int, X: torch.Tensor, adj: torch.Tensor, model_class, num_epochs: int = 100) -> float:
    device = C_0.device
    C_0_new = torch.zeros(C_0.shape, dtype=C_0.dtype, device=device)
    # C_0_new = torch.zeros(C_0.shape, dtype=C_0.dtype)
    for i in range(C_0.shape[0]):
        C_0_new[i][torch.argmax(C_0[i])] = 1

    Lc = C_0_new.T @ L @ C_0_new
    # Wc = (-1 * Lc) * (1 - torch.eye(Lc.shape[0]).cpu())
    Wc = (-1 * Lc) * (1 - torch.eye(Lc.shape[0], device=device))
    Wc[Wc < 0.1] = 0
    Wc = Wc.cpu().detach().numpy()
    Wc = sparse.csr_matrix(Wc)
    Wc = Wc.tocoo()
    # row = torch.from_numpy(Wc.row).to(torch.long)
    # col = torch.from_numpy(Wc.col).to(torch.long)
    # edge_index_coarsen2 = torch.stack([row, col], dim=0)
    # edge_weight = torch.from_numpy(Wc.data)
    row = torch.from_numpy(Wc.row).to(device).long()
    col = torch.from_numpy(Wc.col).to(device).long()
    edge_index_coarsen2 = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(Wc.data).to(device)

    def one_hot(x, class_count):
        return torch.eye(class_count, dtype=C_0.dtype)[x, :].cpu()

    Y = one_hot(labels, num_classes).to(device)
    # print(Y)
    
    num_rows_to_zero = int(0.2 * Y.size(0))

    # Randomly select the indices of rows to zero out
    # torch.manual_seed(42)  # Fixed seed for reproducibility
    indices_to_zero = sample(range(0, Y.size(0)), num_rows_to_zero)

    # Create a mask for training nodes (1 for trainable nodes, 0 for excluded nodes)
    train_mask = torch.ones(Y.size(0), dtype=torch.bool)
    train_mask[indices_to_zero] = False  # Exclude selected rows

    # Set the corresponding rows in Y to zero
    Y[~train_mask] = 0
    # print(Y)
# 
    # The remaining nodes (20%) are for testing
    test_mask = ~train_mask
    check = 0
    try:
        P = torch.linalg.pinv(C_0_new)
    except:
        check = 1
    if check == 0:

        # Y = Y.squeeze(1)  

        labels_coarse = torch.argmax(torch.sparse.mm(torch.Tensor(P).double(), Y.double()).double(), 1)
        Wc = Wc.toarray()


        # model = model_class(X.shape[1], num_classes).to('cpu')
        model = model_class(X.shape[1], num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

        x = sample(range(0, int(k)), k)
        Xt = P @ X.to_dense()

        def train():
            model.train()
            optimizer.zero_grad()
            out = model(torch.Tensor(Xt).to(device), edge_index_coarsen2)
            loss = F.nll_loss(out[x], labels_coarse[x])
            loss.backward()
            optimizer.step()
            return loss

        losses = []
        for epoch in range(num_epochs):
            loss = train()
            losses.append(loss)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch:03d}, loss: {loss:.4f}')

        zz = sample(range(0, int(X.shape[0])), X.shape[0])
        Wc = sparse.csr_matrix(adj.cpu().numpy())
        Wc = Wc.tocoo()
        # row = torch.from_numpy(Wc.row).to(torch.long)
        # col = torch.from_numpy(Wc.col).to(torch.long)
        # edge_index_coarsen = torch.stack([row, col], dim=0)
        # edge_weight = torch.from_numpy(Wc.data) 
        row = torch.from_numpy(Wc.row).to(device).long()
        col = torch.from_numpy(Wc.col).to(device).long()
        edge_index_coarsen = torch.stack([row, col], dim=0)
        edge_weight = torch.from_numpy(Wc.data).to(device)

        pred = model(torch.Tensor(X.to_dense()).to(device), edge_index_coarsen).argmax(dim=1)
        correct = (pred[test_mask] == torch.tensor(labels[test_mask], device=device)).sum().item()
        acc = int(correct) / test_mask.sum().item()
        # print(acc,"hahhhahahha")
        return acc
    else:
        return 0
    


def experiment_deltaC2(
    X_coarse_old: torch.Tensor, X_just_new: torch.Tensor, A_old: torch.Tensor, 
    A_old_new: torch.Tensor, D: torch.Tensor, M1: torch.Tensor, M2: torch.Tensor, 
    M4: torch.Tensor, alpha: float, beta: float, omega: float, 
    zeta: float, C_old: torch.Tensor, delta_CN, delta_CR, k: int, new_nodes: int,
    A_c_surv: torch.Tensor = None
):
    """
    Function to optimize ΔC_N and ΔC_S for the given objective function.

    Args:
        X_coarse_old, X_just_new, A_old, A_old_new, D, M1, M2, M3, M4: Input tensors.
        alpha, beta, omega, zeta: Hyperparameters for the objective function.
        C_old: Tensor representing the current coarse graph embedding.
        k: Dimension of coarse embeddings.
        new_nodes: Number of new nodes.
        A_c_surv: Surviving Coarsened Adjacency from T-1 (optional).

    Returns:
        Optimized ΔC_N and ΔC_S.
    """
    device = C_old.device  # Assume all tensors are already on the same device
    dtype = C_old.dtype
    # Initialize ΔC_N and ΔC_S
    
    

    # Optimizer (SGD)
    eta0 = 0.01  # Learning rate for SGD
    delta_CN=torch.nn.Parameter(delta_CN)
    delta_CR=torch.nn.Parameter(delta_CR)
    optimizer = torch.optim.SGD([delta_CN, delta_CR], lr=eta0, momentum=0.9, nesterov=True, weight_decay=1e-4)
    tol = 1e-6
    thresh = 1e-10

    loss_history = []
    min_epochs = 20
    rel_tol = 1e-4
    patience = 5
    stable_count = 0

    # Fallback if A_c_surv is not provided (should not happen in ACNR correct usage but safe)
    if A_c_surv is None:
        # Fallback to standard C_old^T A_old C_old if A_c_surv not provided (Original logic)
        target_Ac = C_old.T @ A_old @ C_old
    else:
        target_Ac = A_c_surv

    for i in range(200):
        optimizer.zero_grad()  # Clear previous gradients

        # Compute ΔA_ct (difference in adjacency matrices of the coarsened graphs)
        # Note: delta_A_ct here actually calculates A_c_new, not the difference itself.
        # The formula below expands (C_old + dCR)^T A_new (C_old + dCR + dCN...)
        # Wait, the formula:
        # A_ct = (C_new)^T A_new C_new
        #      = [C_old+dCR; dCN]^T [A_old_new M2; M2^T M4] [C_old+dCR; dCN]
        #      ... expansion ...
        # The implementation below:
        delta_A_ct = (((C_old.T @ M2) + (delta_CR.T @ M2) + (delta_CN.T @ M4)) @ delta_CN + ((delta_CR.T @ A_old) + (delta_CN.T @ M2.T)) @ (C_old + delta_CR) + (C_old.T @ A_old @ delta_CR) )        
        
        # NOTE: The variable name `delta_A_ct` in the original code seems to represent the DIFFERENCE A_ct - A_c_old, 
        # or it is missing the C_old.T @ A_old @ C_old term to be the full A_ct.
        # Let's re-verify the expansion:
        # Full A_c_new = (C_old+dCR)^T A_old_new (C_old+dCR)  <-- Top-left block
        #              + (C_old+dCR)^T M2 dCN                <-- Top-right interaction
        #              + dCN^T M2^T (C_old+dCR)              <-- Bottom-left interaction
        #              + dCN^T M4 dCN                        <-- Bottom-right block
        
        # The code calculates:
        # term A: ((C_old.T @ M2) + (delta_CR.T @ M2) + (delta_CN.T @ M4)) @ delta_CN 
        #         = C_old.T M2 dCN + dCR.T M2 dCN + dCN.T M4 dCN
        # term B: ((delta_CR.T @ A_old) + (delta_CN.T @ M2.T)) @ (C_old + delta_CR)
        #         = dCR.T A_old C_old + dCR.T A_old dCR + dCN.T M2.T C_old + dCN.T M2.T dCR
        # term C: (C_old.T @ A_old @ delta_CR)
        #
        # Sum = (C_old.T M2 dCN) + (dCR.T M2 dCN) + (dCN.T M4 dCN) +
        #       (dCR.T A_old C_old) + (dCR.T A_old dCR) + (dCN.T M2.T C_old) + (dCN.T M2.T dCR) +
        #       (C_old.T A_old dCR)
        #
        # This Sum is exactly A_c_new - (C_old.T @ A_old @ C_old).
        # So `delta_A_ct` is indeed A_c_new - A_c_old.
        #
        # For ACNR with deletion, we want to minimize || A_c_new - A_c_surv ||.
        # Since C_old here IS C_old_surv and A_old IS A_old_surv (passed from main),
        # C_old.T @ A_old @ C_old IS A_c_surv.
        #
        # So `delta_A_ct` = A_c_new - A_c_surv.
        # And the objective term is omega * || delta_A_ct ||^2.
        #
        # Wait, if A_c_surv is passed explicitly, does it differ from C_old.T @ A_old @ C_old?
        # In the "Recalibrate Reference" step, we computed A_c_surv = C_old_surv.T @ A_old_surv @ C_old_surv.
        # And we passed C_old_surv and A_old_surv as C_old and A_old to this function.
        # So `C_old.T @ A_old @ C_old` IS ALREADY `A_c_surv`.
        #
        # So Term 1 (Structure Term) is ALREADY minimizing || A_c_new - A_c_surv ||.
        # We don't need to change Term 1 calculation if C_old/A_old are the surviving ones.
        
        # However, Term 5 (Coarse-grain consistency) was:
        # || (C_old+dCR)^T A_old_new (C_old+dCR) - C_old.T @ A_old @ C_old ||
        # Here A_old_new is M1.
        # C_old.T @ A_old @ C_old is A_c_surv.
        # So this term forces the "Refined Survivor Subgraph" to resemble the "Original Survivor Subgraph".
        # This seems correct for ACNR: "Ensure the term forces the new graph to resemble the surviving structure".
        #
        # So actually, if we pass the pruned matrices correctly, the MATH remains the same structure!
        # The key was passing the PRUNED matrices as C_old and A_old.
        #
        # BUT, the user prompt said:
        # "Replace the standard reference A_{c_{t-1}} with the computed A_{c_{t-1}}^{surv}."
        # If we pass pruned C_old/A_old, then implicit A_{c_{t-1}} becomes A_{c_{t-1}}^{surv}.
        #
        # User also said: "Consistency Term Update: Similarly update the third term ... by replacing C_{t-1} with C_{t-1}^{surv}."
        # Since we pass C_old = C_old_surv, this is also handled implicitly.
        #
        # So, the only explicit change might be ensuring we don't accidentally use the un-pruned versions if they were somehow hardcoded or if logic assumed sizes.
        # But wait, `target_Ac` was hardcoded as `C_old.T @ A_old @ C_old`.
        # If I pass `A_c_surv`, I should use it to be explicit and safe.
        
        term1 = omega * torch.norm(delta_A_ct, p='fro')**2

        # Term for ΔC_N normalization constraint
        term2 = torch.norm(delta_CN @ torch.ones((k, 1), dtype=dtype, device=device) - torch.ones((new_nodes, 1), dtype=dtype, device=device), p='fro')**2

        # Term for structural consistency
        term3 = (beta / 2) * torch.norm((C_old + delta_CR) - D @ delta_CN, p='fro')**2

        # Term for feature preservation
        term4 = zeta * torch.norm(delta_CN @ X_coarse_old - X_just_new, p='fro')**2

        # Term for coarse-grain consistency
        # This term minimizes the difference between the "refined" survivor subgraph and the "original" survivor subgraph.
        delta_CR_plus = C_old + delta_CR
        
        # We replace C_old.T @ A_old @ C_old with target_Ac to be explicit (though they should be equal if inputs are correct)
        term5 = (alpha / 2) * torch.norm(
            delta_CR_plus.T @ A_old_new @ delta_CR_plus - target_Ac, 
            p='fro'
        )**2

        # Total loss
        loss = term1 + term2 + term3 + term4 + term5

        # Compute gradients
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([delta_CN, delta_CR], max_norm=5.0)

        # Perform optimization step
        optimizer.step()

        # Apply non-negativity constraint and threshold
        delta_CR.data = torch.clamp(delta_CR.data, min=thresh)
        delta_CN.data = torch.clamp(delta_CN.data, min=thresh)

        # Normalize rows of ΔC_N and ΔC_S
        for j in range(delta_CR.size(0)):
            delta_CR.data[j] /= torch.linalg.norm(delta_CR.data[j], 1) + 1e-10
        for j in range(delta_CN.size(0)):
            delta_CN.data[j] /= torch.linalg.norm(delta_CN.data[j], 1) + 1e-10

        # Track and check for convergence (relative loss change with patience)
        cur_loss = loss.detach().item()
        loss_history.append(cur_loss)
        if i + 1 >= min_epochs:
            if len(loss_history) >= 2:
                rel_change = abs(loss_history[-2] - loss_history[-1]) / (abs(loss_history[-2]) + 1e-8)
                if rel_change < rel_tol:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= patience:
                    print(f"Convergence reached after {i+1} iterations.")
                    break
    delta_CN.requires_grad=False
    delta_CR.requires_grad=False

    return delta_CN, delta_CR, loss_history



def DGC2(alpha1_param: float, beta1_param: float, omega1_param: float, alpha2_param: float, delta_CN: torch.Tensor, delta_CR: torch.Tensor, A_new: torch.Tensor, A_old: torch.Tensor, C_old: torch.Tensor, size_new: int, size_old: int, new_nodes: int, X_new: torch.Tensor, k: int,X_coarse_old: torch.Tensor, A_c_surv: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    M1, M2, M3, M4 = get_block_matrices2(A_old, A_new, size_old, new_nodes)
    X_just_new= get_feature_new_nodes(X_new,new_nodes)
    A_old_new=M1
    D=M2
    tracemalloc.start()
    if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    delta_CN,delta_CR,loss_history = experiment_deltaC2(X_coarse_old, X_just_new,A_old, A_old_new, D, M1, M2, M4, alpha1_param, beta1_param, omega1_param, alpha2_param, C_old, delta_CN, delta_CR, k, new_nodes, A_c_surv)
    current,peak = tracemalloc.get_traced_memory()
    if torch.cuda.is_available():
        peak_gpu = torch.cuda.max_memory_allocated(device="cuda") 
    else:
        peak_gpu = 0
    peak+=peak_gpu
    tracemalloc.stop()
    # delta_C = experiment_deltaC(M2, M3, M4, zita_param, C_old, delta_C, k, new_nodes)
    argmax_before=torch.argmax(C_old, dim=1)
    C_old=C_old+delta_CR
    argmax_after = torch.argmax(C_old, dim=1) 
    C_new = torch.cat((C_old, delta_CN), dim=0)
    different_mappings_count = (argmax_before != argmax_after).sum().item()
    try:
        X_tilde_new = torch.linalg.pinv(C_new)  @  X_new.to_dense()
        status = True
        return X_tilde_new, C_new, peak, status, different_mappings_count, loss_history
    except:
        status = False
        return 0,0,0,status,different_mappings_count, loss_history