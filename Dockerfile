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
COPY store store

# Export ports
EXPOSE 5004

# Start app
#ENTRYPOINT ["gunicorn", "-c", "app/gunicorn.py", "-k", \
# "uvicorn.workers.UvicornWorker", "app.api:app"]

ENTRYPOINT ["gunicorn", "-c", "app/gunicorn.py", "-k", "uvicorn.workers.UvicornWorker", "-b :5004", "app.api:app"]
#to Run execute below , make sure to set environment variables to set prior->
# docker run --env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY \
#  --env MLFLOW_S3_ENDPOINT_URL --env MLFLOW_TRACKING_URI -p 5004:5004 --name iris iris:latest