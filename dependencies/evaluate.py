import pandas as pd
import sys

def evaluate_submission(submission_file: str, ground_truth_file: str):
    """
    Calculates the Manager Prediction Accuracy by comparing a submission file
    against a ground truth file.

    The accuracy is defined as the percentage of correctly predicted managers
    for all non-CEO employees.

    Args:
        submission_file (str): The path to the candidate's submission CSV file.
                               Expected columns: ['employee_id', 'manager_id']
        ground_truth_file (str): The path to the ground truth CSV file.
                                 Expected columns: ['employee_id', 'manager_id']
    """
    # --- 1. Load Data with Error Handling ---
    try:
        print(f"Loading submission file: '{submission_file}'")
        submission_df = pd.read_csv(submission_file)
        
        print(f"Loading ground truth file: '{ground_truth_file}'")
        ground_truth_df = pd.read_csv(ground_truth_file)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}. Please check your file paths.")
        sys.exit(1) # Exit with an error code
    except Exception as e:
        print(f"\n[ERROR] An error occurred while reading the files: {e}")
        sys.exit(1)

    # --- 2. Validate Input DataFrames ---
    required_submission_cols = ['employee_id', 'manager_id']
    required_truth_cols = ['employee_id', 'manager_id']

    if not all(col in submission_df.columns for col in required_submission_cols):
        print(f"\n[ERROR] Submission file is missing required columns. Expected: {required_submission_cols}")
        sys.exit(1)

    if not all(col in ground_truth_df.columns for col in required_truth_cols):
        print(f"\n[ERROR] Ground truth file is missing required columns. Expected: {required_truth_cols}")
        sys.exit(1)

    submission_df = submission_df.rename(columns={'manager_id': 'true_manager_id'})
    ground_truth_df = ground_truth_df.rename(columns={'manager_id': 'predicted_manager_id'})

    # --- 3. Merge and Compare ---
    print("\nComparing submission against ground truth...")
    
    # Merge the two dataframes on the employee's ID
    # 'inner' merge ensures we only evaluate employees present in both files.
    results_df = pd.merge(submission_df, ground_truth_df, on='employee_id', how='inner')
    
    # --- 4. Filter for Evaluable Employees ---
    # The CEO (manager_id = -1) does not have a manager, so we exclude them from the accuracy calculation.
    evaluable_employees_df = results_df[results_df['true_manager_id'] != -1].copy()
    
    if len(evaluable_employees_df) == 0:
        print("\n[WARNING] No non-CEO employees found to evaluate. Accuracy is 0%.")
        return

    # --- 5. Calculate Accuracy ---
    # Create a boolean column to mark correct predictions
    evaluable_employees_df['is_correct'] = (
        evaluable_employees_df['predicted_manager_id'] == evaluable_employees_df['true_manager_id']
    )
    
    # Accuracy is the mean of the boolean (True=1, False=0) column
    accuracy = evaluable_employees_df['is_correct'].sum()/ground_truth_df.shape[0]
    
    correct_predictions = evaluable_employees_df['is_correct'].sum()
    total_predictions = len(evaluable_employees_df)

    # --- 6. Print Report ---
    print("\n--- Evaluation Results ---")
    print(f"Correctly Predicted Managers: {correct_predictions}")
    print(f"Total Employees Evaluated (non-CEO): {total_predictions}")
    print(f"Total Employees Evaluated in Provided Ground Truth File: {ground_truth_df.shape[0]}")

    print("--------------------------------------")
    print(f"Manager Prediction Accuracy: {accuracy:.2%}")
    print("--------------------------------------")

    # --- 7. Save Accuracy to File ---
    # This is for the CI/CD pipeline to read the accuracy value.
    with open("accuracy.txt", "w") as f:
        f.write(f"{accuracy:.2%}")


if __name__ == "__main__":
    # This block allows the script to be run from the command line.
    
    # Check if the correct number of arguments is provided.
    if len(sys.argv) != 3:
        print("\n[Usage Error]")
        print("This script requires two arguments to run from the command line:")
        print("1. The path to your submission file (e.g., 'submission.csv')")
        print("2. The path to the ground truth file (e.g., 'data/ground_truth_managers.csv')")
        print("\nExample: python3 dependencies/evaluate.py submission.csv data/ground_truth_managers.csv")
        sys.exit(1)
        
    submission_path = sys.argv[1]
    truth_path = sys.argv[2]
    
    evaluate_submission(submission_path, truth_path)