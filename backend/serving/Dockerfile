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

# Set PYTHONPATH to include the parent directory of src
ENV PYTHONPATH="/app"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip

# Core Python utilities and basic dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    PyYAML \
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

# ML and AI core
RUN pip install --no-cache-dir \
    torch \
    transformers \
    tiktoken \
    spacy \
    nltk \
    networkx \
    rapidfuzz

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

# Document processing
RUN pip install --no-cache-dir \
    python-docx \
    PyMuPDF \
    PyPDF2 \
    pypdfium2 \
    Pillow \
    ocrmypdf \
    ftfy

# Web scraping and automation
RUN pip install --no-cache-dir \
    beautifulsoup4 \
    selenium \
    fake-useragent \
    feedparser \
    filetype \
    scholarly \
    crawl4ai \
    slack-sdk \
    gliner

# Data visualization
RUN pip install --no-cache-dir \
    seaborn

# Math and optimization
RUN pip install --no-cache-dir \
    ortools

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

# Eval code
RUN pip install --no-cache-dir \
    autoevals

# S3 code
RUN pip install --no-cache-dir \
    aioboto3 \
    termcolor

# Gmail code
RUN pip install --no-cache-dir \
    google-api-python-client \
    google-auth-oauthlib \
    google-auth \
    python-dateutil \
    sentence-transformers

# Eval code
RUN pip install --no-cache-dir \
    umap-learn

# Copy the source code
COPY . /app/src

WORKDIR /app/src

EXPOSE 8000

CMD exec gunicorn serving.main_prod:app \
    --bind :8000 \
    --workers 1 \
    --timeout 300 \
    --worker-class uvicorn.workers.UvicornWorker \
    --threads 8 \
    --access-logfile - \
    --error-logfile - \
    --log-level debug