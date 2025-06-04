# Use Python 3.12 slim as the base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy the entire project directory
COPY . .

RUN apt-get update && apt-get install -y unzip \
    && unzip data.zip -d . \
    && rm data.zip \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies for matplotlib and other libraries
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libatlas-base-dev \
    gfortran \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the script and redirect output to output.txt
CMD ["sh", "-c", "python main.py > output.txt"]