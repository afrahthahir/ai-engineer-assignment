#!/bin/bash

# This script sends a POST request to the local server with the employee and connection data.

# 1. Base64 encode the CSV files
EMPLOYEES_B64=$(base64 -w 0 data/employees.csv)
CONNECTIONS_B64=$(base64 -w 0 data/connections.csv)

# 2. Construct the JSON payload
JSON_PAYLOAD=$(printf '{
  "employees_csv_base64": "%s",
  "connections_csv_base64": "%s"
}' "$EMPLOYEES_B64" "$CONNECTIONS_B64")

# 3. Send the request using curl and save the output to a file
echo "$JSON_PAYLOAD" | curl -X POST \
  -H "Content-Type: application/json" \
  -d @- \
  http://localhost:5001/predict \
  -o curl_response.html

echo "Request sent. Server response saved to curl_response.html"