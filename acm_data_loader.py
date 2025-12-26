import os
import pickle
# import dgl
import torch

class ACMDynamicGraph:
    """
    Loads a pre-processed, dynamic sequence of ACM graphs from saved files.
    """
    def __init__(self, processed_graph_dir='ACM_processed_graphs'):
        """
        Initializes the data loader by loading graph snapshots from disk.

        Args:
            processed_graph_dir (str): The directory containing the saved
                                       'acm_snapshot_{i}.pkl' files.
        """
        self.processed_graph_dir = processed_graph_dir
        self.num_tasks = 6
        self.graphs = self._load_graphs()

    def _load_graphs(self):
        """Loads the 6 pre-processed temporal graph snapshots from disk."""
        graphs = []
        print(f"Loading pre-processed ACM graphs from '{self.processed_graph_dir}'...")
        for i in range(self.num_tasks):
            file_path = os.path.join(self.processed_graph_dir, f'acm_snapshot_{i}.pkl')
            try:
                with open(file_path, 'rb') as f:
                    g = pickle.load(f)
                    graphs.append(g)
                    print(f"  - Loaded graph T={i} with {g.num_nodes()} nodes.")
            except FileNotFoundError:
                print(f"ERROR: Processed graph file not found at '{file_path}'.")
                print("Please run the 'preprocess_acm.py' script first to generate the graph files.")
                raise
        print("All ACM graph snapshots loaded successfully.")
        return graphs

    def get_graph_for_timestep(self, t):
        """
        Returns the graph data for a specific time step.

        Args:
            t (int): The time step (from 0 to 5).

        Returns:
            tuple: A tuple containing:
                   - edge_index (torch.Tensor): The edge index of the graph.
                   - x (torch.Tensor): The node features.
                   - y (torch.Tensor): The node labels.
                   - node_idxs (torch.Tensor): The global node indices.
        """
        if not 0 <= t < self.num_tasks:
            return None, None, None, None
        
        g = self.graphs[t]
        
        if g.num_nodes() == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,0)), torch.empty((0,)), torch.empty((0,))

        edge_src, edge_dst = g.edges()
        edge_index = torch.stack([edge_src, edge_dst], dim=0)

        # Return node_idxs if available, otherwise assume sequential range (fallback)
        if 'node_idxs' in g.ndata:
            node_idxs = g.ndata['node_idxs']
        else:
            node_idxs = torch.arange(g.num_nodes())
            
        return edge_index, g.ndata['x'], g.ndata['y'], node_idxs
