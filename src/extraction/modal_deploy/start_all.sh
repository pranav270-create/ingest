#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

deploy_type=$1

case $deploy_type in
    "all")
        modal deploy "$SCRIPT_DIR/marker_modal.py"
        modal deploy "$SCRIPT_DIR/ocr_modal.py"
        modal deploy "$SCRIPT_DIR/grobid_modal.py"
        ;;
    "marker")
        modal deploy "$SCRIPT_DIR/marker_modal.py"
        ;;
    "ocr")
        modal deploy "$SCRIPT_DIR/ocr_modal.py"
        ;;
    "grobid")
        modal deploy "$SCRIPT_DIR/grobid_modal.py"
        ;;
    *)
        echo "Invalid deployment type. Please use 'all', 'datalab', 'ocr', or 'grobid'"
        exit 1
        ;;
esac
