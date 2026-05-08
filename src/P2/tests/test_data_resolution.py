import pandas as pd
import networkx as nx
import os
from ppi_graph import build_graph, simulate_binding
from data_resolver import resolve_targets, build_disease_graph

def run_resolution_test():
    print("--- STARTING PERSON 2 DATA RESOLUTION VERIFICATION ---")
    
    # 1. Setup Mock Data for Testing
    graph_nodes = {'ABL1', 'PTPN11', 'EGFR', 'TNF'}
    mock_drug_df = pd.DataFrame({
        'drug_name': ['Imatinib', 'Imatinib', 'TestDrug'],
        'gene_symbol': ['abl1', 'SHP2', 'NONEXISTENT']
    })
    
    print("\n[TEST 1] Verifying ID Resolution & Aliases...")
    # Test case-insensitivity (abl1 -> ABL1) and Alias (SHP2 -> PTPN11)
    resolved, actions, affs = resolve_targets('Imatinib', mock_drug_df, graph_nodes)
    
    if 'ABL1' in resolved and 'PTPN11' in resolved:
        print(f"SUCCESS: Resolved targets: {resolved}")
    else:
        print(f"FAIL: Resolution failed. Got: {resolved}")

    print("\n[TEST 2] Verifying Logging of Failures...")
    # Test a drug target that definitely isn't in the graph
    resolve_targets('TestDrug', mock_drug_df, graph_nodes)
    if os.path.exists('failed_mappings.txt'):
        with open('failed_mappings.txt', 'r') as f:
            content = f.read()
            if 'NONEXISTENT' in content:
                print("SUCCESS: 'NONEXISTENT' logged to failed_mappings.txt.")
    else:
        print("FAIL: failed_mappings.txt not created.")

    print("\n[TEST 3] Verifying Disease Graph Perturbation...")
    # Create a small dummy graph for speed
    G0 = nx.Graph()
    G0.add_edge('TNF', 'NODE_B', weight=0.5)
    
    mock_disease_df = pd.DataFrame({
        'disease_name': ["Parkinson's disease"],
        'gene_symbol': ['TNF']
    })
    
    # Build disease graph (should upweight TNF connections)
    G_dis = build_disease_graph(G0, "parkinson", mock_disease_df)
    
    w_before = G0['TNF']['NODE_B']['weight']
    w_after = G_dis['TNF']['NODE_B']['weight']
    
    print(f"Disease Gene (TNF) weight change: {w_before} -> {w_after}")
    if w_after > w_before:
        print("SUCCESS: Disease graph correctly amplified (stressed) connections.")
    else:
        print("FAIL: Disease graph did not increase weights.")

if __name__ == "__main__":
    run_resolution_test()