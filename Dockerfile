# Build frontend assets using a Node builder stage (Tailwind)
FROM node:18-alpine AS node_builder
WORKDIR /build

# Copy package manifest and Tailwind config then install deps
COPY package.json package-lock.json* tailwind.config.js ./

# Copy source CSS and related files needed for build
COPY src ./src

RUN npm ci && npm run build

# Use official Python runtime as base image for the app
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies (include libs required by WeasyPrint & Tesseract OCR)
RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        postgresql-client \
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf-xlib-2.0-0 \
        libffi-dev \
        libxml2 \
        libxslt1.1 \
        libjpeg62-turbo-dev \
        zlib1g-dev \
        pkg-config \
        fonts-dejavu-core \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-ara \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Copy built frontend assets from the Node builder
COPY --from=node_builder /build/src/web/static/css/output.css /app/src/web/static/css/output.css

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV FLASK_APP=src.web.app:app
ENV FLASK_ENV=production

# Run the Flask application using Gunicorn and Render's dynamic PORT
CMD gunicorn src.web.app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
