# AI Engineer Assignment: Build the MLOps Automation Pipeline

## 🎯 Goal
Your mission is to optimize the core script and build a robust CI/CD pipeline for this repository. You will improve the performance of the solution, containerize the application, automate model validation, and set up a continuous deployment process.

## 🚀 Your Tasks

### 1. Task 1: Optimize the Solution Script
**File:** `scripts/solution.py`

**Objective:** The current implementation of `scripts/solution.py` is intentionally inefficient and slow. Your task is to identify performance bottlenecks and optimize the code to make it run significantly faster.

**Requirements:**
-   **Analyze:** Profile the script to find the most time-consuming parts. 
-   **Optimize:** Implement changes to reduce the execution time. Pay close attention to:
    -   The text embedding generation process (`SentenceTransformer`).
    -   Inefficient loops or redundant calculations within the scoring functions.
    -   Opportunities for parallelization or vectorized operations (e.g., using NumPy).
-   **Goal:**  The real time from `time python3 scripts/solution.py` will be used to measure improvements. Optimized solution.py script should run faster when measured as clock time on a 8 core core machine.
---

### 2. Task 2: Optimize the Flask Serving Script
**File:** `serving/serve.py`

**Objective:** The current implementation of `serving/serve.py` which wraps the direct reporting line prediction and reporting line visualization scripts over a REST API. it is intentionally inefficient and slow to respond. Your task is to optimize the `serving/serve.py` to make response to a request made with `./tests/send_request.sh` immediate.

**Requirements:**
-   **Analyze:** Profile the script to find the most time-consuming parts. 
-   **Optimize:** Implement changes to reduce the execution time. Pay close attention to:
    -   Preloading the Model on app start instead of loading it every request.
-   **Goal:** Improvement in time required to execute `./tests/send_request.sh` as measured by `time ./tests/send_request.sh`
---

### 3. Task 3: Create a Dockerfile for the Application
**File:** `Dockerfile`

**Objective:** Create a `Dockerfile` to containerize the application, making it portable and ready for deployment.

**Requirements:**
-   **`Dockerfile`:**
    -   Create a `Dockerfile` that packages the entire application.
    -   **Use a multi-stage build.** This is crucial for creating a lean, production-ready image by separating the build environment from the final runtime environment. This practice leads to smaller image sizes and improved security.
    -   The entry point for the container must be the `serving/serve.py` script.
-   **Local Verification:** You should be able to build the Docker image locally and run it to ensure it works correctly use the `./tests/send_request.sh` file to check if the built image and serving script is working correctly.

---

### 4. Task 4: Implement the Pull Request Validation Workflow
**File:** `.github/workflows/pull_request.yml`

**Objective:** Create a GitHub Actions workflow that automatically evaluates the performance of new code submitted via pull requests to the `main` branch.

**Requirements:**
-   **Trigger:** The workflow must run automatically when a pull request is opened or updated against the `main` branch.
-   **Logic:**
    1.  Run the `scripts/solution.py` from the pull request branch.
    2.  Evaluate the solution's output using `dependencies/evaluate.py` to get the new branch's accuracy.
    3.  Run the `scripts/solution.py` from the `main` branch.
    4.  Evaluate the main branch's solution to get its accuracy.
    5.  Compare the accuracy of the new branch against the `main` branch.
-   **Output:**
    -   Post a comment on the pull request that clearly states the accuracy of both the new branch and the `main` branch.
    -   The comment must indicate whether the new branch's solution is an improvement.
-   **Performance:**
    -   Your workflow should run as fast as possible. Use caching mechanisms for dependencies (e.g., pip packages) to accelerate the setup process in subsequent runs.

---

### 5. Task 5: Implement the Continuous Deployment Workflow
**File:** `.github/workflows/serve.yml`

**Objective:** Create a workflow that automatically builds and pushes the Docker image to a container registry whenever the `main` branch is updated.

**Requirements:**
-   **GitHub Actions Workflow (`.github/workflows/serve.yml`):**
    -   **Trigger:** The workflow must run automatically on every push to the `main` branch.
    -   **Action:** Build a Docker image based on the `Dockerfile` created in Task 3.
    -   **Deployment:** Push the newly built image to a container registry (e.g., GitHub Container Registry), making it ready for deployment.

---

## ✅ Deliverables in Priority
#### Try to complete first 4 steps and 5th step only if you have experience with building cicd workflows. Try to complete the assignment in 2 hours time.
1.  A zip version of your repo with `main` branch containing all your changes.
2.  It shall have an optimized `scripts/solution.py` that meets the performance goal.
3.  It shall have an optimized `serving/serve.py` which responds to the request faster.
4.  It shall have a complete and functional `Dockerfile` that uses a multi-stage build caching pip steps.
5.  It shall have implementations for both `pull_request.yml` and `serve.yml`.

You may use `bash evaluate.sh` script to automate tests of the all deliverables and check the `sample_submission_evaluation.md` file on how your submission will be evaluated.
---

Good luck!
