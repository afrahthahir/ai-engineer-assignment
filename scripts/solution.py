import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import plotly.graph_objects as go
import multiprocessing
import torch # <-- Make sure torch is imported
import os

# --- Configuration ---
# Set the number of processes equal to the number of CPU cores for max parallelization
NUM_PROCESSES = os.cpu_count() or 8 
print(f"INFO: Using {NUM_PROCESSES} processes for parallel embedding.")

print("--- Manager Prediction using Hybrid Scoring (Embeddings + Graph Features) ---")

# --- GLOBAL OPTIMIZATION 1: COMPILED REGEX FOR SENIORITY ---
# Define ranks and compile regex patterns once when the module loads
SENIORITY_RANKS = [
    (r'\b(chief|ceo)\b', 7),
    (r'\b(vp|vice president)\b', 6),
    (r'\b(director|head)\b', 5),
    (r'\b(manager|lead)\b', 4),
    (r'\b(senior|principal|sr\.)\b', 3),
    (r'\b(junior|entry|associate)\b', 1)
]
COMPILED_RANKS = [(re.compile(pattern, re.IGNORECASE), score) for pattern, score in SENIORITY_RANKS]

# --- 1. CONFIGURATION: The Weights ---
WEIGHT_EMBEDDING_SIMILARITY = 1.0
WEIGHT_COMMON_NEIGHBORS = 1.0
WEIGHT_SENIORITY_GAP = 1.0
WEIGHT_LOCATION_MATCH = 1.0


def create_embeddings(model, texts):
    """
    Generates sentence embeddings using parallel processing for speed.
    This is the CRITICAL optimization step.
    """
    print("Generating text embeddings (Optimized with Parallelization)...")
    
    # Start the multi-process pool, assigning each process to a CPU core.
    pool = model.start_multi_process_pool(target_devices=[f'cpu:{i}' for i in range(NUM_PROCESSES)])
    
    # Generate embeddings using the parallel pool
    embeddings = model.encode_multi_process(
        texts, 
        pool, 
        show_progress_bar=True
    )
    
    # Stop the pool and join the processes
    model.stop_multi_process_pool(pool)
    
    return embeddings

# --- 2. DATA LOADING ---
def load_data(employees_path, connections_path):
    """Loads employee and connection data."""
    print("Step 1: Loading data...")
    try:
        employees_df = pd.read_csv(employees_path, engine='python')
        connections_df = pd.read_csv(connections_path)
        print(f"Loaded {len(employees_df)} employees and {len(connections_df)} connections.")
        return employees_df, connections_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure required CSV files are present.")
        return None, None
    

# --- OPTIMIZED SENIORITY FUNCTION ---

def get_seniority_optimized(title):

    """Calculates seniority score using pre-compiled regex and stops on first match."""
    title = str(title).lower()
    score = 2
    # Search for the highest rank first and stop immediately when found
    for pattern, rank_score in COMPILED_RANKS:
        if pattern.search(title):
            return rank_score
    return score

# --- 3. FEATURE ENGINEERING & GRAPH CONSTRUCTION (Optimized)---
def build_graph_with_features(employees_df, connections_df):
    """Builds the graph and enriches it with all necessary node attributes."""
    print("Step 2: Engineering features and building graph...")
    
    print("Generating text embeddings (Optimized with Batching)...")
    
    employees_df['combined_text'] = employees_df['job_title_current'].fillna('') + ". " + employees_df['profile_summary'].fillna('')
    model = SentenceTransformer('all-MiniLM-L6-v2', device=torch.device('cpu'))

    # --- Optimization 2: Batch Encoding ---
    texts = employees_df['combined_text'].tolist()
    # Batch encoding is significantly faster
    embeddings = create_embeddings(model, texts)
    # Create dictionary mapping employee ID to its 1D embedding array
    embedding_dict = dict(zip(employees_df['employee_id'], embeddings))

    # --- Optimization 3: Use optimized seniority function ---
    employees_df['seniority_score'] = employees_df['job_title_current'].apply(get_seniority_optimized)

    print("Constructing NetworkX graph... (Optimized Node Addition)...")
    G = nx.Graph()
    
    node_attributes = employees_df.set_index('employee_id').to_dict('index')

    for node_id, attrs in node_attributes.items():
        attrs['embedding'] = embedding_dict.get(node_id)

        # Convert 1D numpy array back to list for cleaner NetworkX storage
        if isinstance(attrs['embedding'], np.ndarray):
            attrs['embedding'] = attrs['embedding'].tolist()

    # --- Optimization 4: Add nodes and attributes in one step (efficient) ---

    G.add_nodes_from(node_attributes.items())
    G.add_edges_from(connections_df.values)
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


# --- 4. THE INFERENCE ALGORITHM (Optimized) ---
def score_potential_managers(employee_id, G):
    employee_attrs = G.nodes[employee_id]
    employee_seniority = employee_attrs.get('seniority_score', 0)
    

    if employee_seniority >= 7:
        return []
    
    # --- Optimization 5: Pre-process employee embedding ONCE (outside the loop) ---
    employee_embedding = employee_attrs.get('embedding')
    employee_embedding_2d = None
    if employee_embedding is not None:
        # np.atleast_2d ensures the correct (1, N) shape for cosine_similarity
        employee_embedding_2d = np.atleast_2d(employee_embedding)

    neighbors = list(G.neighbors(employee_id))
    if not neighbors:
        return []

    candidates = [n_id for n_id in neighbors if G.nodes[n_id].get('seniority_score', 0) > employee_seniority]
    if not candidates:
        candidates = neighbors

    scored_candidates = []
    for cand_id in candidates:
        cand_attrs = G.nodes[cand_id]
        score = 0

        cand_embedding = cand_attrs.get('embedding')
   
        # --- Optimized Cosine Similarity Calculation ---
        if employee_embedding_2d is not None and cand_embedding is not None:
            # Cand embedding must still be processed inside the loop
            cand_embedding_2d = np.atleast_2d(cand_embedding)
            # Use the pre-processed employee_embedding_2d
            similarity = cosine_similarity(
                employee_embedding_2d,
                cand_embedding_2d
            )[0][0]
            score += similarity * WEIGHT_EMBEDDING_SIMILARITY

        # --- Optimization 6: Common neighbors (Minor: using set intersection is often clearer/faster than list() conversion) ---

        # Note: nx.common_neighbors returns a generator, len(list()) is fine, but set intersection is a standard NetworkX pattern.
        common_neighbors = len(list(nx.common_neighbors(G, employee_id, cand_id)))
        score += common_neighbors * WEIGHT_COMMON_NEIGHBORS

        seniority_gap = cand_attrs.get('seniority_score', 0) - employee_seniority
        if seniority_gap > 0:
            score += (1.0 / seniority_gap) * WEIGHT_SENIORITY_GAP

        if cand_attrs.get('location') == employee_attrs.get('location'):
            score += WEIGHT_LOCATION_MATCH

        scored_candidates.append((score, employee_id, cand_id))

    return scored_candidates

def predict_managers_globally(G):
    all_possible_pairs = []
    print("Step 3: Scoring all possible employee-manager pairs...")
    for emp_id in tqdm(G.nodes(), desc="Scoring Progress"):
        all_possible_pairs.extend(score_potential_managers(emp_id, G))

    all_possible_pairs.sort(key=lambda x: x[0], reverse=True)

    print("\nStep 4: Building hierarchy and preventing cycles...")
    final_predictions = {}
    assigned_employees = set()
    hierarchy_graph = nx.DiGraph()

    for score, emp_id, mgr_id in tqdm(all_possible_pairs, desc="Assigning Managers"):
        if emp_id in assigned_employees:
            continue

        hierarchy_graph.add_edge(emp_id, mgr_id)

        if nx.is_directed_acyclic_graph(hierarchy_graph):
            final_predictions[emp_id] = mgr_id
            assigned_employees.add(emp_id)
        else:
            hierarchy_graph.remove_edge(emp_id, mgr_id)

    return final_predictions


# --- 5. FULL WRAPPER FOR API (Prediction + Visualization) ---

def generate_sunburst_html_in_memory(employees_df, connections_df):
    """
    Runs the full prediction pipeline and generates the Plotly Sunburst HTML 
    string directly in memory (optimized).
    """
    
    print("Step 2-4: Running full prediction pipeline...")
    company_graph = build_graph_with_features(employees_df, connections_df)
    manager_predictions = predict_managers_globally(company_graph)

    # 5. Format predictions (equivalent to reading submission.csv)
    predictions_df = pd.DataFrame(manager_predictions.items(), columns=['employee_id', 'manager_id'])
    
    # Handle unassigned/CEO logic
    predictions_df['manager_id'] = predictions_df['manager_id'].fillna(0).astype(int)
    predictions_df.loc[predictions_df['employee_id'] == 358, 'manager_id'] = -1
    
    # 6. Merge data (logic adapted from visualize_sunburst.py)
    df = pd.merge(employees_df, predictions_df, on='employee_id', how='left')

    # 7. Prepare data for Sunburst
    ids = df['employee_id']
    labels = df['name'].fillna('Unknown')
    # The CEO's manager_id is -1 or 0, which we convert to an empty string for the root.
    parents = df['manager_id'].apply(lambda x: '' if x <= 0 else x) 
    hover_text = df['job_title_current'].fillna('N/A')

    # 8. Create Figure
    print("Step 5: Generating in-memory Plotly HTML...")
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        hovertemplate='<b>%{label}</b><br>Title: %{customdata}<br>ID: %{id}<extra></extra>',
        customdata=hover_text,
        insidetextorientation='radial'
    ))

    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        title=dict(
            text='Interactive Employee Organizational Chart',
            font=dict(size=20),
            x=0.5
        )
    )

    # 9. Return the HTML string directly
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# --- 5. MAIN EXECUTION  (Optimized DataFrame Merge) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--employees_path', default='data/employees.csv')
    parser.add_argument('--connections_path', default='data/connections.csv')
    parser.add_argument('--output_path', default='submission.csv')
    args = parser.parse_args()

    employees, connections = load_data(args.employees_path, args.connections_path)

    if employees is not None:
        company_graph = build_graph_with_features(employees, connections)
        manager_predictions = predict_managers_globally(company_graph)

        print("\nStep 5: Generating Submission File...")

        # --- Optimization 7: Efficient prediction mapping ---
        # Use map function for potentially faster lookup and assignment
        predictions_map = {emp_id: mgr_id for emp_id, mgr_id in manager_predictions.items()}
        submission_df = pd.DataFrame(employees['employee_id'])
        # Map predictions, defaulting to 0 if no manager was assigned
        submission_df['manager_id'] = submission_df['employee_id'].map(predictions_map).fillna(0).astype(int)
        submission_df.loc[submission_df['employee_id'] == 358, 'manager_id'] = -1
        submission_df.to_csv(args.output_path, index=False)
        print(f"\nProcessing complete. Cycle-free submission file saved as '{args.output_path}'.")
