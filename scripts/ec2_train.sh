#!/bin/bash
# EC2 bootstrap: pull processed data from S3, train YOLOv8, push model to S3
set -e

BUCKET="s3://argus-training-data-890615325560-us-east-1-an"

# Install dependencies
pip install ultralytics boto3

# Pull repo
git clone https://github.com/Adnan082/Argus_Target_System.git /home/ec2-user/argus
cd /home/ec2-user/argus

# Pull processed dataset from S3
mkdir -p data/processed
aws s3 sync $BUCKET/processed/ data/processed/

# Run training
python src/training/train.py \
    --data configs/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --name argus-v1 \
    --s3-bucket argus-training-data-890615325560-us-east-1-an

echo "Training complete"
