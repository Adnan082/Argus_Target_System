#!/bin/bash
# EC2 bootstrap: pull xView data from S3, run EDA notebook, push results to GitHub, self-terminate
set -e

BUCKET="s3://argus-training-data-890615325560-us-east-1-an"
REGION="us-east-1"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# Log everything to file — check this if something goes wrong
exec > /home/ec2-user/eda.log 2>&1

echo "=== ARGUS EDA started: $(date) ==="

# Fetch GitHub PAT from Secrets Manager
GITHUB_TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id argus/github-pat \
    --region $REGION \
    --query SecretString \
    --output text)

# Install dependencies
pip install jupyter nbconvert matplotlib pillow numpy boto3

# Clone repo using token for push access
git clone https://Adnan082:${GITHUB_TOKEN}@github.com/Adnan082/Argus_Target_System.git /home/ec2-user/argus
cd /home/ec2-user/argus
git config user.email "adnancheema917@gmail.com"
git config user.name "ARGUS EC2"

# Pull xView data from S3
echo "=== Downloading data from S3: $(date) ==="
mkdir -p data/raw/train_labels data/raw/train_images
aws s3 cp $BUCKET/xview/train_labels.zip data/raw/
aws s3 cp $BUCKET/xview/train_images.zip data/raw/

unzip -q data/raw/train_labels.zip -d data/raw/train_labels
unzip -q data/raw/train_images.zip  -d data/raw/train_images

# Run notebook headlessly — saves all plot outputs inline
echo "=== Running EDA notebook: $(date) ==="
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=1800 \
    notebooks/01_xview_eda.ipynb

# Push notebook with results to GitHub
echo "=== Pushing results to GitHub: $(date) ==="
git add notebooks/01_xview_eda.ipynb
git commit -m "EDA results: xView dataset analysis [automated EC2 run $(date +%Y-%m-%d)]"
git push

echo "=== EDA complete: $(date) ==="

# Self-terminate
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
