#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$1" ]; then
    echo "Running with Uvicorn..."
    source .env
    cd serving
    uvicorn main_prod:app --reload --port 8080 --host 0.0.0.0
else
    echo "Running with Docker..."
    # Check if Dockerfile exists
    if [ ! -f Dockerfile ]; then
        echo "Error: Dockerfile not found"
        exit 1
    fi
    
    # Check if .env.docker exists
    if [ ! -f .env.docker ]; then
        echo "Error: .env.docker not found"
        exit 1
    fi
    
    sudo docker build -t internal_app .
    sudo docker run -p 8080:8080 --env-file .env.docker internal_app
fi