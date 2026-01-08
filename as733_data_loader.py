import os
import pickle
import torch
import dgl

class AS733DynamicGraph:
    """
    Loads pre-processed AS-733 dynamic graphs.
    """
    def __init__(self, processed_graph_dir='AS733_processed_graphs'):
        self.processed_graph_dir = processed_graph_dir
        self.num_tasks = 5
        self.graphs = self._load_graphs()
        
    def _load_graphs(self):
        graphs = []
        print(f"Loading pre-processed AS-733 graphs from '{self.processed_graph_dir}'...")
        for i in range(self.num_tasks):
            file_path = os.path.join(self.processed_graph_dir, f'as733_snapshot_{i}.pkl')
            try:
                with open(file_path, 'rb') as f:
                    g = pickle.load(f)
                    graphs.append(g)
                    print(f"  - Loaded graph T={i} with {g.num_nodes()} nodes.")
            except FileNotFoundError:
                print(f"ERROR: Processed graph file not found at '{file_path}'.")
                print("Please run 'preprocess_as733.py' first.")
                raise
        return graphs

    def get_graph_for_timestep(self, t):
        if not 0 <= t < self.num_tasks:
            return None, None, None, None
        
        g = self.graphs[t]
        
        if g.num_nodes() == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,0)), torch.empty((0,)), torch.empty((0,))

        # DGL to PyG edge index
        edge_src, edge_dst = g.edges()
        edge_index = torch.stack([edge_src, edge_dst], dim=0)

        node_idxs = g.ndata['node_idxs'] if 'node_idxs' in g.ndata else torch.arange(g.num_nodes())
        
        return edge_index, g.ndata['x'], g.ndata['y'], node_idxs

    def get_num_classes(self):
        # Scan all graphs to find max label
        max_label = 0
        for g in self.graphs:
            if 'y' in g.ndata:
                max_label = max(max_label, g.ndata['y'].max().item())
        return max_label + 1

