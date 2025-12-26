import os
import pickle
from collections import defaultdict

import dgl
import numpy as np
import torch
from nltk.tokenize import word_tokenize
import networkx as nx

def create_and_save_dynamic_graphs(raw_data_dir='./', output_dir='../data/ACM_processed_graphs/'):
    """
    Performs the entire preprocessing pipeline once and saves the final
    DGL graph objects to disk.
    """
    # --- Configuration ---
    initial_end_year = 1995
    increment_years = 2
    num_tasks = 6
    
    venue_map = {
        'ai': {'AAAI', 'IJCAI', 'KDD', 'CIKM'},
        'sp': {'ICASSP', 'ICIP'},
        'am': {'SODA', 'STACS'},
        'iss': {'SIGMOD', 'VLDB'}
    }
    class_map = {'ai': 0, 'sp': 1, 'am': 2, 'iss': 3}

    def get_label_and_class(venue_string):
        for class_name, venues in venue_map.items():
            for venue_abbr in venues:
                if venue_abbr in venue_string:
                    return class_map[class_name]
        return -1

    # --- Step 1: Load Raw Data ---
    processed_data_path = os.path.join(raw_data_dir, 'ACM_processed.pickle')
    try:
        with open(processed_data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded '{processed_data_path}'.")
    except FileNotFoundError:
        print(f"ERROR: The file '{processed_data_path}' was not found.")
        raise

    # --- Step 2: Filter data ---
    max_year = initial_end_year + (num_tasks - 1) * increment_years
    print(f"Filtering papers based on specified venues up to year {max_year}...")
    all_items = {}
    for item_id, item in data.items():
        if 'year' in item and item['year'] <= max_year and 'venue' in item:
            label = get_label_and_class(item['venue'])
            if label != -1:
                all_items[item_id] = {
                    'label': label, 'title': item.get('title', ''),
                    'year': item['year'], 'ref': item.get('ref', set())
                }
    print(f"Found {len(all_items)} papers matching the criteria.")

    # --- Step 3: Build Vocabulary ---
    print("Creating token dictionary...")
    token_dic = defaultdict(int)
    for item in all_items.values():
        for token in word_tokenize(item['title'].lower()):
            token_dic[token] += 1
    
    valid_tokens = [token for token, count in token_dic.items() if 5 < count < 5000]
    token_idx = {token: i for i, token in enumerate(valid_tokens)}
    print(f"Created a vocabulary of {len(token_idx)} tokens.")

    for item in all_items.values():
        feature = [0] * len(token_idx)
        for token in word_tokenize(item['title'].lower()):
            if token in token_idx:
                feature[token_idx[token]] += 1
        item['feature'] = feature

    # --- Step 4: Build temporal graphs ---
    print("Building temporal graphs...")
    temporal_graphs = []
    id_master_map = {item_id: i for i, item_id in enumerate(all_items)}
    for t in range(num_tasks):
        current_end_year = initial_end_year + t * increment_years if t > 0 else initial_end_year
        nodes_in_snapshot = {
            item_id for item_id, item_data in all_items.items() 
            if item_data['year'] <= current_end_year
        }
        id_to_local_idx = {item_id: i for i, item_id in enumerate(nodes_in_snapshot)}
        
        g = dgl.DGLGraph()
        if nodes_in_snapshot:
            g.add_nodes(len(nodes_in_snapshot))
            src_edges, dst_edges = [], []
            for item_id, local_idx in id_to_local_idx.items():
                item_year = all_items[item_id]['year']
                for ref_id in all_items[item_id]['ref']:
                    if ref_id in id_to_local_idx and all_items.get(ref_id, {}).get('year', 9999) <= item_year:
                        ref_local_idx = id_to_local_idx[ref_id]
                        src_edges.extend([local_idx, ref_local_idx])
                        dst_edges.extend([ref_local_idx, local_idx])
            if src_edges:
                g.add_edges(src_edges, dst_edges)
            g.ndata['x'] = torch.tensor([all_items[item_id]['feature'] for item_id in id_to_local_idx.keys()], dtype=torch.float32)
            g.ndata['y'] = torch.tensor([all_items[item_id]['label'] for item_id in id_to_local_idx.keys()], dtype=torch.long)
            g.ndata['node_idxs'] = torch.tensor([id_master_map[item_id] for item_id in id_to_local_idx.keys()], dtype=torch.long)
        temporal_graphs.append(g)
        print(f"  - Built graph for T={t} (years <= {current_end_year}) with {g.num_nodes()} nodes.")

    # --- Step 5: Filter for LCC, identify new nodes, and SAVE ---
    print("Filtering, identifying new nodes, and saving graphs...")
    os.makedirs(output_dir, exist_ok=True)
    seen_node_indices = set()
    for i, g in enumerate(temporal_graphs):
        if g.num_nodes() == 0:
            print(f"  - Final graph T={i}: 0 nodes. Saving empty graph.")
            final_g = g
        else:
            nx_g = dgl.to_networkx(g).to_undirected()
            components = list(nx.connected_components(nx_g))
            if not components:
                final_g = g.subgraph([])
            else:
                largest_component_nodes = max(components, key=len)
                final_g = g.subgraph(list(largest_component_nodes))
            
            current_node_indices = set(final_g.ndata['node_idxs'].tolist())
            new_nodes_mask = [1 if node_idx not in seen_node_indices else 0 for node_idx in final_g.ndata['node_idxs'].tolist()]
            final_g.ndata['new_nodes_mask'] = torch.tensor(new_nodes_mask, dtype=torch.bool)
            seen_node_indices.update(current_node_indices)
            print(f"  - Final graph T={i}: {final_g.num_nodes()} nodes ({sum(new_nodes_mask)} new).")

        # Save the final graph object
        graph_path = os.path.join(output_dir, f'acm_snapshot_{i}.pkl')
        with open(graph_path, 'wb') as f:
            pickle.dump(final_g, f)
        print(f"    -> Saved to '{graph_path}'")
        
    print("\nPreprocessing and saving complete!")

if __name__ == '__main__':
    create_and_save_dynamic_graphs()
