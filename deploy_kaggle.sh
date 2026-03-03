#!/bin/bash
# Kaggle Deployment Script for Dynamic Entropy Genuineness Framework

if [ -z "$KAGGLE_API_TOKEN" ]; then
    echo "Error: KAGGLE_API_TOKEN environment variable is not set."
    exit 1
fi

KAG_CLI=$(which kaggle)
if [ -z "$KAG_CLI" ]; then
    echo "Error: 'kaggle' command not found in PATH."
    exit 1
fi

echo "Pushing kernel to Kaggle..."
"$KAG_CLI" kernels push -p .

echo "Deployment complete."
