#!/bin/bash
# EC2 bootstrap: download xView from S3, chip into 640x640 tiles, upload processed dataset to S3
# Launch with: 80GB EBS, IAM role ec2-s3-access
set -e

BUCKET="s3://argus-training-data-890615325560-us-east-1-an"
REGION="us-east-1"

# Capture instance ID before redirecting output
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

exec > /tmp/preprocess.log 2>&1
echo "=== ARGUS Preprocessing started: $(date) ==="

# GitHub PAT from Secrets Manager
GITHUB_TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id argus/github-pat \
    --region $REGION \
    --query SecretString \
    --output text | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

# System installs
dnf install -y git python3-pip p7zip

# Python packages
python3 -m pip install --ignore-installed pillow numpy boto3

# Clone repo
git clone https://Adnan082:${GITHUB_TOKEN}@github.com/Adnan082/Argus_Target_System.git /home/ec2-user/argus
cd /home/ec2-user/argus

# Download raw data from S3
echo "=== Downloading data: $(date) ==="
mkdir -p data/raw/train_labels data/raw/train_images
aws s3 cp $BUCKET/xview/train_labels.zip data/raw/
aws s3 cp $BUCKET/xview/train_images.zip data/raw/

# Extract labels
unzip -q data/raw/train_labels.zip -d data/raw/train_labels

# Extract images (ZIP64 — requires 7za)
/usr/bin/7za x data/raw/train_images.zip -o/home/ec2-user/argus/data/raw/train_images -y

# Free ~30GB before processing
rm data/raw/train_labels.zip data/raw/train_images.zip

# Run preprocessing: convert → stratified split → chip
echo "=== Running preprocessing: $(date) ==="
python3 src/preprocessing/chip_and_split.py \
    --labels data/raw/train_labels/xView_train.geojson \
    --images data/raw/train_images/train_images \
    --output data/processed

# Upload processed dataset to S3
echo "=== Uploading to S3: $(date) ==="
aws s3 sync data/processed/ $BUCKET/processed/ --delete

echo "=== Preprocessing complete: $(date) ==="

# Self-terminate
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
