from flask import Flask, request, jsonify, send_file
import base64
import os
import sys
import io
import pandas as pd


# Ensure the scripts directory is accessible for importing solution.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))


app = Flask(__name__)

# --- API ROUTE (INSTANT RESPONSE) ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts employee and connection data as base64 encoded CSVs,
    runs the hierarchy prediction, and returns the sunburst visualization.
    """

    # CRITICAL FIX: Import inside the function to prevent startup crashes
    from solution import generate_sunburst_html_in_memory
    
    data = request.get_json()

    if not data or 'employees_csv_base64' not in data or 'connections_csv_base64' not in data:
        return jsonify({"error": "Missing required fields: employees_csv_base64, connections_csv_base64"}), 400

    try:
        # 1. Decode base64 strings directly into in-memory buffers (OPTIMIZATION)
        employees_io = io.BytesIO(base64.b64decode(data['employees_csv_base64']))
        connections_io = io.BytesIO(base64.b64decode(data['connections_csv_base64']))

        # 2. Read the data directly from the in-memory buffers (OPTIMIZATION)
        employees_df = pd.read_csv(employees_io)
        connections_df = pd.read_csv(connections_io)
        
        # 3. Run the FULL pipeline using the direct, in-memory function call
        # This replaces subprocess calls, file I/O, and separate prediction/visualization steps.
        html_content = generate_sunburst_html_in_memory(employees_df, connections_df)

        # 4. Return the HTML content with the correct MIME type
        return html_content, 200, {'Content-Type': 'text/html'}

    except Exception as e:
        app.logger.error(f"Error processing dynamic data: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5001)