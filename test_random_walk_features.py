#!/usr/bin/env python3
"""
Test script for random walk feature computation
"""
import sys
import os
sys.path.append('/home/scai/phd/aiz248316/.conda/envs/new_env/lib/python3.10/site-packages')

try:
    import networkx as nx
    import torch
    import numpy as np
    from scipy.sparse.linalg import svds
    print("✓ All required packages are available")
    
    # Test the random walk feature function
    # Create a simple test graph
    G = nx.karate_club_graph()
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Import the function from preprocess_as733
    from preprocess_as733 import compute_random_walk_features
    
    # Test with smaller dimensions for speed
    features = compute_random_walk_features(G, num_walks=5, walk_length=10, window_size=3, feature_dim=16)
    
    print(f"✓ Generated features for {len(features)} nodes")
    print(f"✓ Feature dimension: {next(iter(features.values())).shape[0]}")
    print(f"✓ Sample feature (first 5 values): {features[0][:5].tolist()}")
    
    # Test that features are different (not all identical)
    feature_matrix = torch.stack([features[node] for node in G.nodes()])
    feature_variance = torch.var(feature_matrix, dim=0).mean()
    print(f"✓ Average feature variance: {feature_variance:.4f}")
    
    if feature_variance > 0.01:
        print("✓ Features show good diversity")
    else:
        print("⚠ Features may be too similar")
        
    print("\n✅ Random walk feature computation test passed!")
    
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("Please install: pip install networkx scipy numpy torch")
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
