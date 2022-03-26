# Base image
FROM python:3.8-buster

# Install dependencies
WORKDIR mlops
COPY setup.py setup.py
COPY requirements.txt requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e . --no-cache-dir \
    && apt-get purge -y --auto-remove gcc build-essential

# Copy
COPY iris iris
COPY app app
COPY data data
COPY config config
COPY stores stores

# Export ports
EXPOSE 5004

# Start app
ENTRYPOINT ["gunicorn", "-c", "app/gunicorn.py", "-k", "uvicorn.workers.UvicornWorker", "app.api:app"]