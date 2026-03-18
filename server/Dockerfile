FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for pyserial (USB/serial support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    udev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -e .

# Expose Flask port
EXPOSE 11434

# Run the Flask app
CMD ["uvicorn", "tv_commercial_detector.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "11434"]
