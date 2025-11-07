import pandas as pd
import networkx as nx
import sys

def find_hierarchy_cycles(submission_file: str):
    """
    Loads a manager submission file and reports any circular reporting structures.
    """
    print(f"--- Checking for cycles in '{submission_file}' ---")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(submission_file)
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {submission_file}")
        sys.exit(1)

    # --- 2. Build a Directed Graph ---
    # A directed graph is perfect because the manager relationship is one-way.
    G = nx.DiGraph()
    
    print("Building a directed graph of the hierarchy...")
    for _, row in df.iterrows():
        employee_id = row['employee_id']
        manager_id = row['manager_id']

        # We only add an edge if there is a valid manager
        # CEOs (-1) and unassigned managers (0 or NaN) don't create edges upward.
        if pd.notna(manager_id) and manager_id not in [-1, 0]:
            # The edge goes FROM the employee TO their manager
            G.add_edge(employee_id, int(manager_id))

    # --- 3. Find and Report Cycles ---
    try:
        # networkx has a built-in function to find all simple cycles
        cycles = list(nx.simple_cycles(G))
        
        if not cycles:
            print("\n[SUCCESS] No cycles were found in the hierarchy. It is a valid tree structure.")
        else:
            print(f"\n[WARNING] Found {len(cycles)} cycle(s) in the hierarchy:")
            print("-------------------------------------------------")
            for i, cycle in enumerate(cycles, 1):
                # Format the cycle for readability, e.g., "123 -> 456 -> 123"
                path = " -> ".join(map(str, cycle))
                print(f"  Cycle {i}: {path} -> {cycle[0]}")
            print("-------------------------------------------------")
            print("These cycles must be broken for the hierarchy to be valid.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during cycle detection: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\n[Usage]")
        print("python3 dependencies/find_cycles.py <submission_file.csv>")
        print("\nExample: python3 dependencies/find_cycles.py submission.csv")
        sys.exit(1)

    submission_path = sys.argv[1]
    find_hierarchy_cycles(submission_path)
