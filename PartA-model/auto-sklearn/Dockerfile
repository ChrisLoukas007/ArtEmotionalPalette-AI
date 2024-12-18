FROM python:3.9-slim

# Update and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    python3-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir numpy==1.23.5 \
    scipy==1.9.1 \
    scikit-learn==0.24.2 \
    matplotlib \
    seaborn \
    auto-sklearn==0.14.7 \
    pandas==1.3.5

# Create a writable directory
RUN mkdir -p /app/tmp

WORKDIR /app
CMD ["bash"]
