from __future__ import annotations
import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.utils import to_dense_adj

# Automatically detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_dir: str = "/tmp/CS") -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Load the Cora dataset and return feature matrix X, adjacency matrix adj, and labels.
    
    Args:
        data_dir (str): Directory to download/load the dataset.
        
    Returns:
        tuple: Feature matrix X, adjacency matrix adj, and labels.
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset = Planetoid(root=data_dir, name='CS')
    adj = to_dense_adj(dataset[0].edge_index)[0].to(device)  # Move adjacency matrix to the device
    labels = dataset[0].y.numpy()  # Labels remain on CPU
    X = dataset[0].x.to_dense().to(device)  # Move feature matrix to the device
    return X, adj, labels

def preprocess_data(X: torch.Tensor, adj: torch.Tensor, labels: np.ndarray, subset_ratio: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Preprocess the data by selecting a subset based on subset_ratio.
    
    Args:
        X (torch.Tensor): Feature matrix.
        adj (torch.Tensor): Adjacency matrix.
        labels (np.ndarray): Labels array.
        subset_ratio (float): Ratio of the subset to select.
        
    Returns:
        tuple: Subset feature matrix X, subset adjacency matrix adj, and subset labels.
    """
    if not (0 < subset_ratio <= 1):
        raise ValueError("subset_ratio must be between 0 and 1")
    
    nn = int(subset_ratio * X.shape[0])
    X = X[:nn, :].to(device)  # Move subset to the device
    adj = adj[:nn, :nn].to(device)  # Move subset to the device
    labels = labels[:nn]  # Labels remain on CPU
    return X, adj, labels

def get_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute the Laplacian matrix from the adjacency matrix.
    
    Args:
        adj (torch.Tensor): Adjacency matrix.
        
    Returns:
        torch.Tensor: Laplacian matrix.
    """
    b = torch.ones(adj.shape[0], device=device)  # Ensure vector is on the device
    return torch.diag(adj @ b) - adj

def get_block_matrices(A_old: torch.Tensor, A_new: torch.Tensor, size_old: int, new_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate block matrices for incremental graph updates.
    
    Args:
        A_old (torch.Tensor): Old adjacency matrix.
        A_new (torch.Tensor): New adjacency matrix.
        size_old (int): Size of the old graph.
        new_nodes (int): Number of new nodes.
        
    Returns:
        tuple: Block matrices M1, M2, M3, and M4.
    """
    M1 = A_old.to(device)  # Move to device
    M2 = A_new[:size_old, size_old:].to(device)
    M3 = torch.transpose(M2, 0, 1).to(device)
    M4 = A_new[size_old:, size_old:].to(device)
    return M1, M2, M3, M4

def get_block_matrices2(A_old: torch.Tensor, A_new: torch.Tensor, size_old: int, new_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate block matrices for incremental graph updates.
    
    Args:
        A_old (torch.Tensor): Old adjacency matrix.
        A_new (torch.Tensor): New adjacency matrix.
        size_old (int): Size of the old graph.
        new_nodes (int): Number of new nodes.
        
    Returns:
        tuple: Block matrices M1, M2, M3, and M4.
    """
    M1 = A_new[:size_old, :size_old].to(device)  # Move to device
    M2 = A_new[:size_old, size_old:].to(device)
    M3 = torch.transpose(M2, 0, 1).to(device)
    M4 = A_new[size_old:, size_old:].to(device)
    return M1, M2, M3, M4

def get_feature_new_nodes(X_new: torch.Tensor, new_nodes: int) -> torch.Tensor:
    """
    Extract features for the new nodes from the feature matrix.
    
    Args:
        X_new (torch.Tensor): Feature matrix.
        new_nodes (int): Number of new nodes.
        
    Returns:
        torch.Tensor: Features of the new nodes.
    """
    X_N = X_new[-new_nodes:].to(device)  # Move to device
    return X_N  