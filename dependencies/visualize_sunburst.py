import pandas as pd
import plotly.graph_objects as go
import argparse

def visualize_sunburst_hierarchy(employees_file, submission_file, output_html_file):
    """
    Creates an interactive sunburst chart to visualize the employee hierarchy.
    """
    print("--- Starting Sunburst Hierarchy Visualization ---")

    # --- 1. Load Data ---
    try:
        print("Loading data files...")
        employees_df = pd.read_csv(employees_file)
        gt_df = pd.read_csv(submission_file)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all required CSV files are present.")
        return

    # --- 2. Prepare Data for Sunburst ---
    print("Preparing data for the sunburst chart...")
    
    # Merge the data to have all information in one place
    df = pd.merge(employees_df, gt_df, on='employee_id')

    # The sunburst trace requires specific data fields: ids, labels, and parents.
    
    # The 'ids' are the unique identifiers for each employee
    ids = df['employee_id']
    
    # The 'labels' are the names that will be displayed on the chart
    labels = df['name']
    
    # The 'parents' are the IDs of the managers.
    # The CEO's manager_id is -1, which we'll convert to an empty string for the root of the chart.
    parents = df['manager_id'].apply(lambda x: '' if x == -1 else x)
    
    # We can also add custom data to show on hover
    hover_text = df['job_title_current']

    # --- 3. Create and Save the Figure ---
    print(f"Generating and saving the interactive plot to '{output_html_file}'...")
    
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

    fig.write_html(output_html_file)
    print("\n--- Visualization Complete! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--employees_path', default='data/employees.csv')
    parser.add_argument('--submission_path', default='submission.csv')
    parser.add_argument('--output_path', default='employee_sunburst.html')
    args = parser.parse_args()

    visualize_sunburst_hierarchy(
        employees_file=args.employees_path,
        submission_file=args.submission_path,
        output_html_file=args.output_path
    )