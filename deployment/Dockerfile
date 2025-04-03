FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for the web interface
EXPOSE 8501

# Set up a volume for persistent data
VOLUME /app/datasets

# Run the application
ENTRYPOINT ["streamlit", "run", "advanced_web_interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
