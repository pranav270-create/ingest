# Remove comments and empty lines, then convert to comma-separated KEY=VALUE pairs
env_vars=$(grep -v '^#' .env | grep -v '^$' | sed 's/^export //' | xargs | sed 's/ /,/g')
echo "Deploying with environment variables configured..."

# First command line is the name of the deployment
if [ -z "$1" ]; then
    echo "Error: Deployment name not provided."
    echo "Usage: $0 <deployment-name>"
    exit 1
fi

gcloud run deploy $1 --source . --memory=4Gi --min-instances=2 --region us-central1 --allow-unauthenticated --timeout 3600s --set-env-vars="$env_vars"