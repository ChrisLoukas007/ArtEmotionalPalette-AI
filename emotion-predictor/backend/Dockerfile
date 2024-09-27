FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and install packages one by one
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi==0.68.0 && \
    pip install --no-cache-dir uvicorn==0.15.0 && \
    pip install --no-cache-dir numpy==1.21.0 && \
    pip install --no-cache-dir opencv-python-headless==4.5.3.56 && \
    pip install --no-cache-dir scikit-learn==1.5.1 && \
    pip install --no-cache-dir tensorflow==2.16.0rc0 && \
    pip install --no-cache-dir python-multipart==0.0.5

COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]