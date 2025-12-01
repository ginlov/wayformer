# Dockerfile
FROM oel20/cvrunner-base-amd:latest

# Copy both files if they exist (wildcard makes them optional)
RUN echo "Test echo"

COPY requirements.txt* poetry.lock* .
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -U pip && \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        echo "No requirements.txt found, skipping."; \
    fi

# Copy and install poetry.lock if it exists
RUN if [ -f poetry.lock ]; then \
        pip install --no-cache-dir poetry && \
        poetry install --no-dev; \
    else \
        echo "No poetry.lock found, skipping."; \
    fi

COPY . /workspace

# Set working directory
WORKDIR /workspace

# Set default entrypoint to cvrunner CLI
ENTRYPOINT ["cvrunner"]
