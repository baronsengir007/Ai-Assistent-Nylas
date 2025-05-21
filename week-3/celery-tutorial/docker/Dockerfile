FROM python:3.11-slim

WORKDIR /app

# Copy only the application files
COPY celery_app.py .
COPY tasks.py .
COPY client.py .
# Add other python files if you have them, or a requirements.txt
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Install dependencies directly for now
RUN pip install --no-cache-dir celery redis

# Verify celery is installed and in PATH
RUN which celery || echo "Celery not found in PATH"

# Set the Python path
ENV PYTHONPATH=/app

# Use the full path to celery
CMD ["/usr/local/bin/celery", "-A", "tasks", "worker", "--loglevel=info"]