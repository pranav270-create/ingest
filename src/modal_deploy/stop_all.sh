#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

deploy_type=$1

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it first:"
    echo "  - macOS: brew install jq"
    echo "  - Ubuntu/Debian: sudo apt-get install jq"
    echo "  - CentOS/RHEL: sudo yum install jq"
    exit 1
fi

get_app_id() {
    local description=$1
    modal app list --json | jq -r ".[] | select(.Description == \"$description\" and .State == \"deployed\") | .\"App ID\"" | head -n 1
}

case $deploy_type in
    "all")
        marker_id=$(get_app_id "marker-modal")
        ocr_id=$(get_app_id "ocr-modal")
        grobid_id=$(get_app_id "grobid-modal")
        moondream_id=$(get_app_id "moondream-modal")
        mineru_id=$(get_app_id "mineru-modal")
        colpali_id=$(get_app_id "colpali-modal")
        
        [ ! -z "$marker_id" ] && modal app stop "$marker_id"
        [ ! -z "$ocr_id" ] && modal app stop "$ocr_id"
        [ ! -z "$grobid_id" ] && modal app stop "$grobid_id"
        [ ! -z "$moondream_id" ] && modal app stop "$moondream_id"
        [ ! -z "$mineru_id" ] && modal app stop "$mineru_id"
        [ ! -z "$colpali_id" ] && modal app stop "$colpali_id"
        ;;
    "marker")
        app_id=$(get_app_id "marker-modal")
        [ ! -z "$app_id" ] && modal app stop "$app_id"
        ;;
    "ocr")
        app_id=$(get_app_id "ocr-modal")
        [ ! -z "$app_id" ] && modal app stop "$app_id"
        ;;
    "grobid")
        app_id=$(get_app_id "grobid-modal")
        [ ! -z "$app_id" ] && modal app stop "$app_id"
        ;;
    "moondream")
        app_id=$(get_app_id "moondream-modal")
        [ ! -z "$app_id" ] && modal app stop "$app_id"
        ;;
    "mineru")
        app_id=$(get_app_id "mineru-modal")
        [ ! -z "$app_id" ] && modal app stop "$app_id"
        ;;
    "colpali")
        app_id=$(get_app_id "colpali-modal")
        [ ! -z "$app_id" ] && modal app stop "$app_id"
        ;;
    *)
        echo "Invalid deployment type. Please use 'all', 'marker', 'ocr', 'grobid', 'moondream', 'mineru', or 'colpali'"
        exit 1
        ;;
esac
