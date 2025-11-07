import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import math # Needed for trigonometry
import argparse

def create_radial_layout(employees_df, gt_df):
    """
    Computes node positions for a radial/sunburst-like layout based on the org hierarchy.
    
    Returns:
        A dictionary of positions {node_id: (x, y)}
    """
    print("Building organizational hierarchy for layout...")
    
    # Find the CEO (root of the hierarchy)
    try:
        ceo_id = gt_df[gt_df['manager_id'] == -1]['employee_id'].iloc[0]
    except IndexError:
        print("ERROR: CEO not found in ground_truth_managers.csv. Cannot create layout.")
        return {}

    # Create a map of managers to their direct reports for quick traversal
    reports_map = gt_df[gt_df['manager_id'] != -1].groupby('manager_id')['employee_id'].apply(list).to_dict()

    # --- Step 1: Determine the level of each employee using Breadth-First Search (BFS) ---
    levels = {ceo_id: 0}
    positions = {ceo_id: (0, 0)}
    queue = [ceo_id]
    
    head = 0
    while head < len(queue):
        manager_id = queue[head]
        head += 1
        
        for report_id in reports_map.get(manager_id, []):
            if report_id not in levels:
                levels[report_id] = levels[manager_id] + 1
                queue.append(report_id)

    # --- Step 2: Recursively calculate angular positions ---
    def set_node_positions(manager_id, start_angle, end_angle):
        """Assigns positions to reports within a given angular wedge."""
        reports = reports_map.get(manager_id, [])
        if not reports:
            return

        angle_step = (end_angle - start_angle) / len(reports)
        
        for i, report_id in enumerate(reports):
            # Calculate the angle for this specific report
            angle = start_angle + (i + 0.5) * angle_step
            
            # Radius is based on the employee's level in the org
            level = levels.get(report_id, 0)
            radius = level * 10  # The '10' is an arbitrary step size for visual spacing
            
            # Convert polar coordinates (radius, angle) to Cartesian (x, y)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            positions[report_id] = (x, y)
            
            # Recurse to place this report's own team members
            set_node_positions(report_id, start_angle + i * angle_step, start_angle + (i + 1) * angle_step)

    # Start the recursive placement from the CEO, who owns the full circle (0 to 2*pi radians)
    print("Calculating radial positions for all nodes...")
    set_node_positions(ceo_id, 0, 2 * math.pi)
    
    return positions


def visualize_employee_network_radial(employees_file, connections_file, submission_file, output_html_file):
    """
    Creates an interactive Plotly visualization of the employee network
    with a custom radial layout based on the organizational hierarchy.
    """
    print("--- Starting Radial Network Visualization ---")
    
    # --- 1. Load Data ---
    try:
        print("Loading data files...")
        employees_df = pd.read_csv(employees_file)
        connections_df = pd.read_csv(connections_file)
        gt_df = pd.read_csv(submission_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all required CSV files are present.")
        return

    # --- 2. Build the FULL Graph to get all nodes and edges ---
    print("Building full network graph from connections data...")
    G = nx.from_pandas_edgelist(connections_df, 'employee_id_a', 'employee_id_b')

    # Add any employees who might not have connections but are in the employee list
    for emp_id in employees_df['employee_id']:
        if emp_id not in G:
            G.add_node(emp_id)

    # Add node attributes from the employees dataframe
    for _, row in employees_df.iterrows():
        node = row['employee_id']
        if node in G:
            G.nodes[node]['name'] = row['name']
            G.nodes[node]['title'] = row['job_title_current']
            G.nodes[node]['summary'] = row['profile_summary']

    # --- 3. Generate the Custom Radial Layout ---
    pos = create_radial_layout(employees_df, gt_df)
    
    # Handle nodes that might be in the graph but not the hierarchy (orphans)
    # Place them in a default location to avoid errors
    for node in G.nodes():
        if node not in pos:
            pos[node] = (0,0) # Place orphans at the center

    # --- 4. Create Plotly Traces (same as before, but using the new 'pos') ---
    print("Creating Plotly traces for the graph...")
    
    # Trace for ALL connections (gray lines)
    edge_x, edge_y = [], []
    for edge in G.edges():
        # Ensure both nodes in the edge have a position
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='General Connection')

    # Trace for ground truth manager connections (red lines)
    gt_edge_x, gt_edge_y = [], []
    for _, row in gt_df.iterrows():
        emp = row['employee_id']
        mgr = row['manager_id']
        if mgr != -1 and emp in pos and mgr in pos:
            x0, y0 = pos[emp]
            x1, y1 = pos[mgr]
            gt_edge_x.extend([x0, x1, None])
            gt_edge_y.extend([y0, y1, None])

    gt_edge_trace = go.Scatter(
        x=gt_edge_x, y=gt_edge_y,
        line=dict(width=2, color='red'), # Make manager lines stand out
        hoverinfo='none',
        mode='lines',
        name='True Manager Link')

    # Trace for employee nodes
    node_x, node_y = [], []
    node_text = []
    node_colors = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_info = f"<b>{G.nodes[node].get('name', '')}</b><br>ID: {node}<br>Title: {G.nodes[node].get('title', 'N/A')}"
        node_text.append(node_info)
        # Color nodes by their organizational level
        level = pos.get(node, (0,0)) # Use position to infer level for color
        color_val = math.sqrt(level[0]**2 + level[1]**2) # Distance from center
        node_colors.append(color_val)

    # --- CORRECTED SECTION ---
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_colors,
            size=7,
            colorbar=dict(
                thickness=15,
                xanchor='left',
                # This is the corrected part. 'title' is a dict, not a string.
                title=dict(
                    text='Seniority<br>(Distance from Center)',
                    side='right'
                )
            ),
            line_width=1))
    # --- END OF CORRECTION ---

    # --- 5. Create and Save the Figure ---
    print(f"Generating and saving the interactive plot to '{output_html_file}'...")
    fig = go.Figure(data=[edge_trace, gt_edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Employee Network with Hierarchical Radial Layout',
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    legend=dict(x=0, y=1))
                    )
    
    fig.write_html(output_html_file)
    print("\n--- Visualization Complete! ---")
    print(f"'{output_html_file}' has been created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--employees_path', default='data/employees.csv')
    parser.add_argument('--connections_path', default='data/connections.csv')
    parser.add_argument('--submission_path', default='submission.csv')
    parser.add_argument('--output_path', default='employee_network_radial.html')
    args = parser.parse_args()

    visualize_employee_network_radial(
        employees_file=args.employees_path,
        connections_file=args.connections_path,
        submission_file=args.submission_path,
        output_html_file=args.output_path
    )
