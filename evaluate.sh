#!/usr/bin/env bash

# This script automates the evaluation of the 5 MLOps pipeline tasks.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Helper Functions ---
info() {
    echo "INFO: $1"
}

# --- Task 1: Performance Check for solution.py ---
info "--- Task 1: Performance Check (scripts/solution.py) ---"
info "Running the solution script and measuring execution time..."
time python3 scripts/solution.py
info "The script should complete in under 30 seconds on any mid tier machine"
echo "Press [Enter] to continue..."
read -r _

# --- Task 2: Performance Check for serve.py ---
info "\n--- Task 2: Performance Check (serving/serve.py) ---"
info "Starting the Flask server in the background..."
python3 serving/serve.py &
SERVER_PID=$!

info "Waiting for the server to start (60 seconds)..."
sleep 60 # will be adjusted if your app cannot be ready to serve in 60 seconds.

info "Measuring response time using './tests/send_request.sh'..."
time ./tests/send_request.sh
info "Request sent. Server response saved to 'curl_response.html'."

info "Stopping the Flask server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
info "Server stopped."
echo "Press [Enter] to continue..."
read -r _
rm -f curl_response.html

# --- Task 3: Docker Build and Verification ---
info "\n--- Task 3: Docker Build and Verification ---"
info "Building the Docker image..."
docker build -t reporting-line-prediction-service-image .

info "Cleaning up existing container if it exists..."
sudo docker rm -f reporting-line-prediction-service &>/dev/null || true

info "Running the Docker container in the background..."
sudo docker run -d -p 5001:5001 --name reporting-line-prediction-service reporting-line-prediction-service-image

info "Waiting for the container to start (10 seconds)..."
sleep 10

info "Sending a test request to the containerized service..."
./tests/send_request.sh

info "Request sent. Server response saved to 'curl_response.html'."
info "Please open 'curl_response.html' in a browser to verify the visualization."
info "Cleaning up the running container..."
sudo docker stop reporting-line-prediction-service &>/dev/null || true
sudo docker rm reporting-line-prediction-service &>/dev/null || true
echo "Press [Enter] to continue..."
read -r _

# --- Task 4: Pull Request Workflow Simulation ---
info "\n--- Task 4: Pull Request Workflow Simulation ---"
MAIN_BRANCH="main"
REMOTE="origin"

info "Cleaning up potentially leftover branches and PRs from previous runs..."
git checkout $MAIN_BRANCH
git branch -D location_match_tweak &>/dev/null || true
git branch -D embedding_similarity_tweak &>/dev/null || true
git push $REMOTE --delete location_match_tweak &>/dev/null || true
git push $REMOTE --delete embedding_similarity_tweak &>/dev/null || true
gh pr list --head location_match_tweak --json number -q '.[] | .number' | xargs -r -I{} gh pr close {}
gh pr list --head embedding_similarity_tweak --json number -q '.[] | .number' | xargs -r -I{} gh pr close {}
info "Cleanup complete."

info "Creating a branch with a performance DECREASE ('location_match_tweak')..."
git checkout -b location_match_tweak
sed -i 's/WEIGHT_LOCATION_MATCH = 0.0/WEIGHT_LOCATION_MATCH = 1.0/' scripts/solution.py
git commit -am "Tweak: Set WEIGHT_LOCATION_MATCH to 1.0"
git push -f $REMOTE location_match_tweak
gh pr create --base $MAIN_BRANCH --head location_match_tweak --title "Feat: Tweak location match weight" --body "This PR should result in higher accuracy."

info "Creating a branch with a performance INCREASE ('embedding_similarity_tweak')..."
git checkout $MAIN_BRANCH
sed -i 's/WEIGHT_LOCATION_MATCH = 0.0/WEIGHT_LOCATION_MATCH = 5.0/' scripts/solution.py # Revert previous change
git checkout -b embedding_similarity_tweak
sed -i 's/WEIGHT_EMBEDDING_SIMILARITY = 1.0/WEIGHT_EMBEDDING_SIMILARITY = 0.0/' scripts/solution.py
sed -i 's/WEIGHT_COMMON_NEIGHBORS = 1.0/WEIGHT_COMMON_NEIGHBORS = 2.0/' scripts/solution.py
git commit -am "Tweak: Increase embedding and common neighbors weights"
git push -f $REMOTE embedding_similarity_tweak
gh pr create --base $MAIN_BRANCH --head embedding_similarity_tweak --title "Feat: Tweak embedding similarity and common neighbors weights" --body "This PR should result in lower accuracy."

info "Two pull requests have been created."
info "Please go to the GitHub repository to verify the CI pipeline comments on both pull requests."
info "The 'location_match_tweak' PR should show higher performance."
info "The 'embedding_similarity_tweak' PR should show lower performance."
info "Once verified, merge the 'embedding_similarity_tweak' PR to trigger the CD workflow."
echo "Press [Enter] to continue after merging the successful PR..."
read -r _

# --- Task 5: Continuous Deployment (CD) Workflow Verification ---
info "\n--- Task 5: CD Workflow Verification ---"
info "After merging the PR, the 'serve' workflow should have been triggered."
info "Please go to the 'Actions' tab in your GitHub repository."
info "Verify that the workflow ran successfully and pushed a new Docker image to the container registry."

echo ""
info "Evaluation script finished."
git checkout $MAIN_BRANCH
