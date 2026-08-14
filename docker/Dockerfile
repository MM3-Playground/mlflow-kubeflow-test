FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        git-annex && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir datalad

COPY runtime-bootstrap.sh /usr/local/bin/runtime-bootstrap
RUN chmod +x /usr/local/bin/runtime-bootstrap

WORKDIR /workspace
