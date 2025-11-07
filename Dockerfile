# ----------------------------------------------------------------------
# STAGE 1: Dependency Installation & Setup
# ----------------------------------------------------------------------

# TODO: Use an official Python runtime as a parent image
FROM python:3.10-slim AS base

# Install system dependencies needed for some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# TODO: Set the working directory in the container
WORKDIR /usr/src/app

# TODO: Copy the dependencies file to the working directory
COPY requirements.txt .

# TODO: Install any needed packages specified in requirements.txt
# Install dependencies, including gunicorn for production serving
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# ----------------------------------------------------------------------
# STAGE 2: Application Setup and Execution
# ----------------------------------------------------------------------

# Copy application source code (serving, scripts, and data)
# TODO: Copy the rest of the application's code
COPY . .

# Expose the port the server runs on
EXPOSE 5001

# TODO: Command to run the application
# Use Gunicorn to run the optimized Flask app in a production environment
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "3", "serving.serve:app"]