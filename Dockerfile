FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Copy requirements and install dependencies (no-cache to keep image smaller)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir wandb kubernetes

# Copy source code
COPY . .

# Non-root user is optional; keep default user for compatibility with CUDA images.

# Recommended to set WANDB API key via your CI or runtime environment rather than baking it into the image.
ENV PYTHONUNBUFFERED=1

CMD ["python", "train_coreweave.py"]