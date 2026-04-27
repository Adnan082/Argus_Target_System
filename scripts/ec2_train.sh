#!/bin/bash
# EC2 bootstrap: pull processed dataset from S3, train YOLOv8, push model to S3
# Launch with: Deep Learning Base AMI (Amazon Linux 2023), g4dn.xlarge, 100GB EBS, IAM role ec2-s3-access
exec > /tmp/train.log 2>&1
set -e
echo "=== ARGUS Training started: $(date) ==="

BUCKET="s3://argus-training-data-890615325560-us-east-1-an"
REGION="us-east-1"

# Capture instance ID
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

# GitHub PAT from Secrets Manager
GITHUB_TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id argus/github-pat \
    --region $REGION \
    --query SecretString \
    --output text | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

# Install dependencies (Deep Learning Base AMI has Python3 + CUDA but no conda envs)
pip3 install ultralytics boto3 --quiet

# Clone repo
git clone https://Adnan082:${GITHUB_TOKEN}@github.com/Adnan082/Argus_Target_System.git /home/ec2-user/argus
cd /home/ec2-user/argus

# Pull processed dataset from S3
echo "=== Downloading processed dataset: $(date) ==="
mkdir -p data/processed
aws s3 sync $BUCKET/processed/ data/processed/

# Run training
echo "=== Running training: $(date) ==="
python3 src/training/train.py \
    --data data/processed/dataset.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 32 \
    --workers 4 \
    --device 0 \
    --name argus-v1 \
    --s3-bucket $BUCKET

echo "=== Training complete: $(date) ==="

# Self-terminate
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
