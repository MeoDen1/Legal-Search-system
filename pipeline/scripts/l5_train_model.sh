#!/bin/bash

KAGGLE_NOTEBOOKS_PATH=$1
DOWNLOAD_PATH=$2
ENV_PATH=$3

# Export environments
export $(grep -Ev '^\s*#.*|^\s*$' $ENV_PATH | xargs)

cd $KAGGLE_NOTEBOOKS_PATH

echo "Kaggle | Pushing notebook..."
kaggle kernels push

NOTEBOOK_ID="$KAGGLE_USERNAME/$(jq -r .id kernel-metadata.json | cut -d/ -f2)"

echo "Kaggle | Waiting for notebook to finish..."

while true; do
    STATUS=$(kaggle kernels status "$NOTEBOOK_ID" | sed -E 's/.*KernelWorkerStatus\.([A-Z]+).*/\1/')
    
    echo "Kaggle | Current status: $STATUS"

    if [[ "$STATUS" == "COMPLETE" ]]; then
        echo "Kaggle | Notebook completed."
        break
    fi

    if [[ "$STATUS" == "ERROR" ]]; then
        echo "Kaggle | Notebook failed."
        exit 1
    fi

    sleep 10
done

echo "Kaggle | Downloading output..."
mkdir -p "$DOWNLOAD_PATH"

kaggle kernels output "$NOTEBOOK_ID" -p "$DOWNLOAD_PATH"
