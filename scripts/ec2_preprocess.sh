#!/bin/bash
# EC2 bootstrap: pull data from S3, preprocess, push back to S3
set -e

BUCKET="s3://argus-training-data-890615325560-us-east-1-an"

# Install dependencies
pip install ultralytics numpy pillow boto3 geopandas rasterio

# Pull repo
git clone https://github.com/Adnan082/Argus_Target_System.git /home/ec2-user/argus
cd /home/ec2-user/argus

# Pull raw data from S3
mkdir -p data/raw
aws s3 cp $BUCKET/xview/train_images.zip data/raw/
aws s3 cp $BUCKET/xview/train_labels.zip data/raw/
aws s3 cp $BUCKET/xview/val_images.zip data/raw/

# Unzip
unzip -q data/raw/train_images.zip -d data/raw/train_images
unzip -q data/raw/train_labels.zip -d data/raw/train_labels
unzip -q data/raw/val_images.zip -d data/raw/val_images

# Convert labels to YOLO format
python src/preprocessing/xview_converter.py \
    --labels data/raw/train_labels/xView_train.geojson \
    --images data/raw/train_images \
    --output data/processed/labels/train

# Upload processed data back to S3
aws s3 sync data/processed/ $BUCKET/processed/

echo "Preprocessing complete"
