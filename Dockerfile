# ----------------------------------------------------------------------
# STAGE 1: BUILDER
# Installs dependencies using PyTorch's CPU wheels index for speed/size.
# This layer is cached and is only rebuilt if requirements.txt changes.
# ----------------------------------------------------------------------
FROM python:3.10-slim-bullseye AS builder

# Install necessary build tools (like build-essential for compiling wheels) and git.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies.
# KEY FIX: Using --extra-index-url to prioritize PyTorch's CPU wheels index.
# This drastically reduces the size and time by avoiding the large CUDA libraries.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -r requirements.txt \
    gunicorn \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Explicitly download and cache the ML model for the runner stage.
# This prevents the server from crashing or hanging on the first request.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# ----------------------------------------------------------------------
# STAGE 2: RUNNER (Final, minimal image)
# Copies only the necessary files for deployment to create a minimal image.
# ----------------------------------------------------------------------
FROM python:3.10-slim-bullseye AS runner

WORKDIR /app

# Copy executable scripts (like 'gunicorn', 'pytest') to the PATH.
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy installed Python site-packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the rest of the application code, data, and cached model files
COPY . .

# Set up the network port
EXPOSE 5001

# Use gunicorn as the production-ready server
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "serving.serve:app"]
