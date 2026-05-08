import sys
import os
import networkx as nx
from ppi_graph import build_graph, simulate_binding

def run_ppi_verification(csv_path):
    print("--- STARTING PERSON 2 INFRASTRUCTURE VERIFICATION ---")
    
    # 1. VERIFY GRAPH LOADING (Prompt 1.1)
    # Check if the graph loads and reports correct node/edge counts
    print(f"\n[TEST 1] Loading graph from: {csv_path}")
    try:
        G0 = build_graph(csv_path)
        print(f"SUCCESS: Loaded {len(G0.nodes)} nodes and {len(G0.edges)} edges.")
    except Exception as e:
        print(f"FAIL: build_graph crashed. Error: {e}")
        return

    # 2. VERIFY NORMALIZATION (Prompt 1.1)
    # Ensure weights are 0-1, not the raw 0-1000 from STRING
    edge_sample = list(G0.edges(data=True))[0]
    weight = edge_sample[2].get('weight', 0)
    print(f"\n[TEST 2] Checking weight normalization...")
    if 0.0 <= weight <= 1.0:
        print(f"SUCCESS: Sample weight is {weight} (Normalized).")
    else:
        print(f"FAIL: Weight {weight} is outside [0, 1] range.")

    # 3. VERIFY DEEP COPY INTEGRITY (Prompt 1.2)
    # This ensures drug simulations don't corrupt the baseline G0
    print(f"\n[TEST 3] Verifying Deep Copy (G0 Protection)...")
    test_targets = ['ABL1'] # Assuming ABL1 is in your CSV
    G_drug, failed = simulate_binding(G0, test_targets)
    
    # Find a common edge to test
    if not G_drug.edges:
        print("SKIP: No edges in G_drug to test.")
    else:
        u, v = list(G_drug.edges())[0]
        original_val = G0[u][v]['weight']
        G_drug[u][v]['weight'] = 9.99 # Corrupt the copy
        
        if G0[u][v]['weight'] == original_val:
            print("SUCCESS: G0 remained unchanged after G_drug modification.")
        else:
            print("CRITICAL FAIL: G0 was modified! Deep copy logic is broken.")

    # 4. VERIFY BINDING LOGIC (Prompt 1.3)
    # Check if weights actually reduced for an inhibitory drug
    print(f"\n[TEST 4] Verifying Inhibitory Attenuation...")
    # Check a neighbor of a target
    if test_targets[0] in G0:
        neighbor = list(G0.neighbors(test_targets[0]))[0]
        w_before = G0[test_targets[0]][neighbor]['weight']
        w_after = G_drug[test_targets[0]][neighbor]['weight']
        print(f"Target {test_targets[0]} -> Neighbor {neighbor}")
        print(f"Weight change: {w_before:.4f} -> {w_after:.4f}")
        if w_after < w_before:
            print("SUCCESS: Binding simulation correctly attenuated the edge.")
        else:
            print("FAIL: Binding simulation did not reduce weight.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ppi_infrastructure.py <path_to_ppi_csv>")
    else:
        run_ppi_verification(sys.argv[1])