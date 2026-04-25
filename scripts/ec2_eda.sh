#!/bin/bash
set -e

BUCKET="s3://argus-training-data-890615325560-us-east-1-an"
REGION="us-east-1"

# Get instance ID before redirecting output
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

exec > /tmp/eda.log 2>&1
echo "=== ARGUS EDA started: $(date) ==="

# Fetch GitHub PAT
GITHUB_TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id argus/github-pat \
    --region $REGION \
    --query SecretString \
    --output text)

# System installs
dnf install -y git python3-pip p7zip

# Python packages — ignore-installed avoids conflict with rpm-managed packages
python3 -m pip install --ignore-installed jupyter nbconvert matplotlib pillow numpy boto3

# Clone repo
git clone https://Adnan082:${GITHUB_TOKEN}@github.com/Adnan082/Argus_Target_System.git /home/ec2-user/argus
cd /home/ec2-user/argus
git config user.email "adnancheema917@gmail.com"
git config user.name "ARGUS EC2"

# Download from S3
echo "=== Downloading data: $(date) ==="
mkdir -p data/raw/train_labels data/raw/train_images
aws s3 cp $BUCKET/xview/train_labels.zip data/raw/
aws s3 cp $BUCKET/xview/train_images.zip data/raw/

# Extract labels (small — unzip handles it fine)
unzip -q data/raw/train_labels.zip -d data/raw/train_labels

# Extract images (15GB ZIP64 — needs 7za, full path for root's PATH)
/usr/bin/7za x data/raw/train_images.zip -o/home/ec2-user/argus/data/raw/train_images -y

# Delete zips to free ~30GB before running notebook
rm data/raw/train_labels.zip data/raw/train_images.zip

# Run EDA notebook
echo "=== Running EDA: $(date) ==="
/usr/local/bin/jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=1800 \
    notebooks/01_xview_eda.ipynb

# Push results to GitHub
echo "=== Pushing to GitHub: $(date) ==="
git add notebooks/01_xview_eda.ipynb
git commit -m "EDA results: xView dataset analysis [automated EC2 run $(date +%Y-%m-%d)]"
git push

echo "=== EDA complete: $(date) ==="

# Self-terminate
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
