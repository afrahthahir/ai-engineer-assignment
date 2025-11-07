import os
import subprocess
import time
import base64
import json
from urllib import request as urlrequest
import pytest

def test_solution_script_runs():
    """
    Tests if the solution.py script runs without errors and produces a submission file.
    """
    # Define the paths for the test data and output
    test_data_dir = 'data'
    output_file = 'test_submission.csv'

    # Ensure the output file doesn't exist before running the script
    if os.path.exists(output_file):
        os.remove(output_file)

    # Run the solution.py script as a subprocess
    result = subprocess.run(
        ['python', 'scripts/solution.py',
         '--employees_path', os.path.join(test_data_dir, 'employees.csv'),
         '--connections_path', os.path.join(test_data_dir, 'connections.csv'),
         '--output_path', output_file],
        capture_output=True,
        text=True
    )

    # Check if the script ran successfully
    assert result.returncode == 0, f"solution.py failed with error: {result.stderr}"

    # Check if the output file was created
    assert os.path.exists(output_file), "solution.py did not create the output file."

    # Clean up the created file
    os.remove(output_file)

def test_serving_script():
    """
    Tests if the serving.py script runs and responds to a request.
    """
    # Start the server as a background process
    server_process = subprocess.Popen(['python', 'serving/serve.py'])
    time.sleep(2)  # Give the server a moment to start

    try:
        # Read and encode the data
        with open('data/employees.csv', 'rb') as f:
            employees_b64 = base64.b64encode(f.read()).decode('utf-8')
        with open('data/connections.csv', 'rb') as f:
            connections_b64 = base64.b64encode(f.read()).decode('utf-8')

        # Prepare the request
        data = json.dumps({
            'employees_csv_base64': employees_b64,
            'connections_csv_base64': connections_b64
        }).encode('utf-8')
        req = urlrequest.Request("http://localhost:5001/predict", data=data, headers={'Content-Type': 'application/json'})

        # Send the request and check the response
        with urlrequest.urlopen(req) as response:
            assert response.status == 200
            assert 'text/html' in response.headers['Content-Type']
            html = response.read()
            assert b'<html>' in html.lower()
            
    except Exception as e:
        pytest.fail(f"An exception occurred during the test: {e}")

    finally:
        # Stop the server
        server_process.terminate()
        server_process.wait()