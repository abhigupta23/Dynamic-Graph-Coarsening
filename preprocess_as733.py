import os
import glob
import pickle
import networkx as nx
import torch
import dgl
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

def compute_random_walk_features(G, num_walks=10, walk_length=20, window_size=5, feature_dim=128):
    """
    Compute random walk-based node features using a simplified approach.
    
    Args:
        G: NetworkX graph
        num_walks: Number of random walks per node
        walk_length: Length of each random walk
        window_size: Context window size for co-occurrence
        feature_dim: Dimension of output features
    
    Returns:
        dict: {node_id: feature_tensor}
    """
    if G.number_of_nodes() == 0:
        return {}
    
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Step 1: Generate random walks
    walks = []
    for node in nodes:
        for _ in range(num_walks):
            walk = [node]
            current = node
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = np.random.choice(neighbors)
                walk.append(current)
            walks.append(walk)
    
    # Step 2: Build co-occurrence matrix efficiently
    vocab_size = len(nodes)
    cooc_matrix = np.zeros((vocab_size, vocab_size))
    
    for walk in walks:
        for i, center_node in enumerate(walk):
            center_idx = node_to_idx[center_node]
            # Define context window
            start = max(0, i - window_size)
            end = min(len(walk), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:  # Skip the center node itself
                    context_node = walk[j]
                    context_idx = node_to_idx[context_node]
                    cooc_matrix[center_idx][context_idx] += 1
    
    # Step 3: Apply PMI (Pointwise Mutual Information) transformation
    total_count = np.sum(cooc_matrix)
    if total_count == 0:
        # Fallback to degree-based features if no co-occurrences
        features = {}
        for node in nodes:
            degree = G.degree(node)
            # Create feature based on degree and neighbor degrees
            neighbor_degrees = [G.degree(neighbor) for neighbor in G.neighbors(node)]
            avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0
            
            # Create a simple feature vector
            feature = torch.zeros(feature_dim)
            feature[0] = degree / 100.0  # Normalize degree
            feature[1] = avg_neighbor_degree / 100.0
            feature[2] = len(neighbor_degrees) / 100.0
            
            # Add some structural diversity
            for i in range(3, min(feature_dim, len(neighbor_degrees) + 3)):
                if i-3 < len(neighbor_degrees):
                    feature[i] = neighbor_degrees[i-3] / 100.0
            
            features[node] = feature
        return features
    
    # Calculate PMI: PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
    row_sums = np.sum(cooc_matrix, axis=1)
    col_sums = np.sum(cooc_matrix, axis=0)
    
    # Avoid division by zero
    row_sums = np.maximum(row_sums, 1e-10)
    col_sums = np.maximum(col_sums, 1e-10)
    
    pmi_matrix = np.zeros_like(cooc_matrix)
    for i in range(vocab_size):
        for j in range(vocab_size):
            if cooc_matrix[i][j] > 0:
                p_ij = cooc_matrix[i][j] / total_count
                p_i = row_sums[i] / total_count
                p_j = col_sums[j] / total_count
                pmi_matrix[i][j] = max(0, np.log(p_ij / (p_i * p_j)))  # Positive PMI
    
    # Step 4: Dimensionality reduction using SVD
    from scipy.sparse.linalg import svds
    
    try:
        # Use sparse SVD for efficiency
        k = min(feature_dim, min(pmi_matrix.shape) - 1)
        if k > 0:
            U, sigma, _ = svds(pmi_matrix, k=k, random_state=42)
            # Sort by singular values (descending)
            idx = np.argsort(sigma)[::-1]
            U = U[:, idx]
            
            # Create feature dictionary
            features = {}
            for i, node in enumerate(nodes):
                features[node] = torch.tensor(U[i], dtype=torch.float32)
        else:
            raise ValueError("SVD dimension too small")
            
    except Exception as e:
        print(f"  SVD failed, using degree-based features: {e}")
        # Fallback to degree-based features
        features = {}
        for node in nodes:
            degree = G.degree(node)
            neighbor_degrees = [G.degree(neighbor) for neighbor in G.neighbors(node)]
            avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0
            
            feature = torch.zeros(feature_dim)
            feature[0] = degree / 100.0
            feature[1] = avg_neighbor_degree / 100.0
            feature[2] = len(neighbor_degrees) / 100.0
            
            for i in range(3, min(feature_dim, len(neighbor_degrees) + 3)):
                if i-3 < len(neighbor_degrees):
                    feature[i] = neighbor_degrees[i-3] / 100.0
            
            features[node] = feature
    
    return features

def preprocess_as733(raw_dir='as733_raw', output_dir='AS733_processed_graphs', num_snapshots=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Get list of files
    files = sorted(glob.glob(os.path.join(raw_dir, 'as*.txt')))
    if not files:
        print(f"No files found in {raw_dir}")
        return

    # Select evenly spaced snapshots
    total_files = len(files)
    step = max(1, total_files // num_snapshots)
    selected_files = [files[i] for i in range(0, total_files, step)][:num_snapshots]
    
    print(f"Selected {len(selected_files)} snapshots: {[os.path.basename(f) for f in selected_files]}")

    node_id_map = {} # Global mapping of Original ID -> 0..N
    next_id = 0
    
    # We need a consistent global feature space if possible, or just random features per node.
    # For dynamic graphs where nodes appear/disappear, we usually assign a permanent ID.
    
    # To ensure consistent features for the same node across time, we generate them lazily.
    # Now using random walk-based features instead of random initialization
    global_features = {} 
    feature_dim = 128
    
    # For labels, we can generate them per snapshot (dynamic communities) or globally.
    # Since DGC evaluates clustering/classification, per-snapshot communities make sense as "current ground truth".
    
    for t, file_path in enumerate(selected_files):
        print(f"Processing T={t}: {os.path.basename(file_path)}")
        
        # Read graph
        G_nx = nx.read_edgelist(file_path, comments='#', nodetype=int)
        
        # Compute random walk features for this snapshot
        print(f"  Computing random walk features...")
        snapshot_features = compute_random_walk_features(G_nx, feature_dim=feature_dim)
        
        # Update global ID map and features
        local_nodes = list(G_nx.nodes())
        for node in local_nodes:
            if node not in node_id_map:
                node_id_map[node] = next_id
                # Use random walk-based feature for this node
                if node in snapshot_features:
                    global_features[next_id] = snapshot_features[node]
                else:
                    # Fallback: create a simple degree-based feature
                    degree = G_nx.degree(node)
                    feature = torch.zeros(feature_dim)
                    feature[0] = degree / 100.0
                    global_features[next_id] = feature
                next_id += 1
                
        # Remap to global 0..N indices for DGL
        # Note: The DGL graph for timestep t will have N_t nodes. 
        # But we need to track their "global identity" to handle alignment in main.py.
        # The existing code expects `node_idxs` in `ndata` to be the global IDs.
        
        # We'll re-label the NX graph to 0..V_t-1 for DGL creation, but keep track of global IDs.
        # Sort nodes to ensure deterministic ordering
        sorted_nodes = sorted(local_nodes)
        node_to_local = {node: i for i, node in enumerate(sorted_nodes)}
        
        # Create DGL graph
        # Relabel NX graph to local 0..K indices
        G_nx_relabelled = nx.relabel_nodes(G_nx, node_to_local)
        g_dgl = dgl.from_networkx(G_nx_relabelled)
        
        # Assign Features
        # Get global ID for each local node i
        global_ids_list = [node_id_map[original_node] for original_node in sorted_nodes]
        
        features_list = [global_features[gid] for gid in global_ids_list]
        g_dgl.ndata['x'] = torch.stack(features_list)
        
        # Assign Global IDs (for alignment)
        g_dgl.ndata['node_idxs'] = torch.tensor(global_ids_list, dtype=torch.long)
        
        # Assign Labels (Community Detection)
        # Use greedy modularity
        try:
            communities = greedy_modularity_communities(G_nx)
            # Create a map node -> community_id
            node_to_comm = {}
            for cid, comm in enumerate(communities):
                for node in comm:
                    node_to_comm[node] = cid
            
            # If some nodes missed (isolated?), assign -1 or 0
            labels_list = [node_to_comm.get(original_node, 0) for original_node in sorted_nodes]
            
        except Exception as e:
            print(f"  Community detection failed: {e}. Assigning random labels.")
            labels_list = np.random.randint(0, 5, size=g_dgl.num_nodes())

        g_dgl.ndata['y'] = torch.tensor(labels_list, dtype=torch.long)
        
        # Save
        save_path = os.path.join(output_dir, f'as733_snapshot_{t}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(g_dgl, f)
            
        print(f"  Saved T={t} with {g_dgl.num_nodes()} nodes, {g_dgl.num_edges()} edges.")

if __name__ == '__main__':
    preprocess_as733()

