FROM python:3.10

WORKDIR /app

RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

RUN APT_INSTALL="apt-get install -y --no-install-recommends --allow-unauthenticated" \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    build-essential \
    libboost-all-dev \
    python3-dev \
    python3-distutils \
    python3-tk \
    python3-pip \
    cmake \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    libldap2-dev \
    libsasl2-dev \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libgl1 \
    libxrender1 \
    libglib2.0-0 \
    scrot \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set PYTHONPATH to include both src and serving directories
ENV PYTHONPATH="/app"

RUN pip install --upgrade pip

# Core Python utilities and basic dependencies
RUN pip install --no-cache-dir \
    numpy \
    python-dotenv \
    aiofiles \
    backoff \
    tenacity \
    loguru \
    joblib \
    tqdm \
    urllib3

# HTTP and API clients
RUN pip install --no-cache-dir \
    requests \
    aiohttp \
    httpx \
    fastapi \
    uvicorn \
    gunicorn

# Database and storage
RUN pip install --no-cache-dir \
    sqlalchemy \
    mysql-connector-python \
    aiomysql \
    boto3

# ML and AI models/services
RUN pip install --no-cache-dir \
    anthropic \
    openai \
    cohere \
    voyageai \
    google-generativeai \
    mistral_common \
    sentence-transformers \
    fastembed \
    qdrant-client \
    instructor

# Security and authentication
RUN pip install --no-cache-dir \
    python-jwt \
    python-jose \
    python-jose[cryptography] \
    passlib \
    python-multipart

# Cron jobs
RUN pip install --no-cache-dir \
    fastapi-utils \
    typing-inspect

# Google Cloud
RUN pip install --no-cache-dir \
    aiohttp \
    cloud-sql-python-connector \
    google-auth \
    google-cloud-storage \
    pg8000 \
    psycopg2-binary
    

# Copy the source code
COPY . .

WORKDIR /app

# print pwd
RUN pwd

EXPOSE 8080

CMD exec gunicorn serving.main_prod:app \
    --bind :8080 \
    --workers 1 \
    --timeout 300 \
    --worker-class uvicorn.workers.UvicornWorker \
    --threads 8 \
    --access-logfile - \
    --error-logfile - \
    --log-level debug