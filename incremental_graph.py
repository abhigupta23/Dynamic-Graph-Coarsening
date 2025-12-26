
import torch
from torch_geometric.datasets import Planetoid,Coauthor
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.utils import subgraph
import os

class IncrementalGraph:
    def __init__(self, initial_nodes=1000, increment_nodes=100, device=None):
        """
        Args:
            initial_nodes (int): Number of initial nodes to include in the subgraph.
            increment_nodes (int): Number of nodes to add in each increment.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.initial_nodes = initial_nodes
        self.increment_nodes = increment_nodes
        self.current_end = initial_nodes

        data_dir = "D:\Abhishek\DGC_15_11_2024"
        os.makedirs(data_dir, exist_ok=True)

        # Determine the device (use GPU if available)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the Cora dataset
        self.dataset = Planetoid(root=data_dir, name='Cora')
        # self.dataset = NodePropPredDataset(root=data_dir,name='ogbn-arxiv')
        self.data = self.dataset[0].to(self.device)
        print(self.data.num_nodes)

        torch.manual_seed(42)

        # Initialize node selection
        self.node_indices = torch.arange(self.data.num_nodes, device=self.device)

        # Initialize the subgraph with the first initial_nodes nodes
        self.sub_edge_index, self.sub_edge_attr, self.sub_x, self.sub_y = self.create_subgraph(self.initial_nodes)

    def create_subgraph(self, max_nodes):
        """
        Args:
            max_nodes (int): Maximum node index to include in the subgraph.

        Returns:
            tuple: Edge index, edge attributes, feature matrix, and labels of the subgraph.
        """
        mask = (self.data.edge_index[0] < max_nodes) & (self.data.edge_index[1] < max_nodes)
        edge_index = self.data.edge_index[:, mask].to(self.device)
        edge_attr = self.data.edge_attr[mask].to(self.device) if self.data.edge_attr is not None else None
        node_indices = torch.arange(max_nodes, device=self.device)
        x = self.data.x[node_indices]
        y = self.data.y[node_indices]

        # Ensure the subgraph includes all nodes up to max_nodes
        isolated_nodes = torch.ones(max_nodes, dtype=torch.bool, device=self.device)
        isolated_nodes[edge_index[0]] = False
        isolated_nodes[edge_index[1]] = False
        isolated_node_indices = torch.nonzero(isolated_nodes).flatten()
        isolated_edges = torch.stack([isolated_node_indices, isolated_node_indices], dim=0)

        edge_index = torch.cat([edge_index, isolated_edges], dim=1)

        return edge_index, edge_attr, x, y

    def add_nodes(self):
        """
        Returns:
            tuple: Updated edge index, feature matrix, edge attributes, labels, and the new end index.
        """
        print('Value of current_end in add_nodes function:', self.current_end)
        new_end = self.current_end
        new_edge_index, new_edge_attr, new_x, new_y = self.create_subgraph(new_end)

        print('Shape of new_x is', new_x.shape)
        print(f"Adding nodes up to {new_end}")

        self.current_end = new_end
        self.sub_edge_index = new_edge_index
        self.sub_edge_attr = new_edge_attr
        self.sub_x = new_x
        self.sub_y = new_y

        return new_edge_index, new_x, new_edge_attr, new_y, self.current_end

    def get_next_subgraph(self):
        """
        Returns:
            tuple: Edge index, feature matrix, and labels of the new subgraph.
        """
        if self.current_end > self.data.num_nodes:
            return None, None, None

        if self.current_end == self.initial_nodes:
            edge_index, x, edge_attr, y = self.sub_edge_index, self.sub_x, self.sub_edge_attr, self.sub_y
            self.current_end += self.increment_nodes
        else:
            edge_index, x, edge_attr, y, self.current_end = self.add_nodes()
            self.current_end += self.increment_nodes

        return edge_index, x, y

# Sample check function to display shapes
def check(edge_index, x, y, device=None):
    """
    Args:
        edge_index (torch.Tensor): Edge index of the subgraph.
        x (torch.Tensor): Feature matrix of the subgraph.
        y (torch.Tensor): Labels of the subgraph.
        device (str): Device to use ('cuda' or 'cpu').
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = edge_index.to(device)
    x = x.to(device)
    y = y.to(device)

    print(f"Running on {device}")
    print("Edge Index Shape:", edge_index.shape)
    print("Feature Vector Shape:", x.shape)
    print("Labels Shape:", y.shape)



def flip_ones_to_zeros(matrix, percentage, device=None):
    """
    Randomly converts a percentage of ones in a binary square matrix to zeros symmetrically.
    
    Args:
        matrix (torch.Tensor): A binary square matrix containing only 0s and 1s.
        percentage (float): The percentage of ones to flip to zeros (0-100).
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        torch.Tensor: The modified matrix.
    """
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")
    
    if matrix.size(0) != matrix.size(1):
        raise ValueError("Matrix must be square for symmetrical flipping.")
    
    # Determine the device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    matrix = matrix.to(device)
    
    # Find the indices of ones in the upper triangle (excluding the diagonal)
    ones_indices = (torch.triu(matrix, diagonal=1) == 1).nonzero(as_tuple=False)
    
    # Determine the number of pairs to flip
    num_ones = ones_indices.size(0)
    num_to_flip = int(num_ones * (percentage / 100))
    
    if num_to_flip == 0:
        return matrix  # No change if percentage is too small
    
    # Randomly select indices to flip
    flip_indices = ones_indices[torch.randperm(num_ones, device=device)[:num_to_flip]]
    
    # Create a copy of the matrix to modify
    modified_matrix = matrix.clone()
    
    # Flip the selected ones to zeros symmetrically
    for idx in flip_indices:
        i, j = idx.tolist()
        modified_matrix[i, j] = 0
        modified_matrix[j, i] = 0
    
    return modified_matrix