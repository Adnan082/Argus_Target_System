#!/bin/bash
# EC2 bootstrap: pull xView data from S3, launch JupyterLab for EDA
set -e

BUCKET="s3://argus-training-data-890615325560-us-east-1-an"

pip install jupyter matplotlib pillow numpy boto3

# Pull repo
git clone https://github.com/Adnan082/Argus_Target_System.git /home/ec2-user/argus
cd /home/ec2-user/argus

# Pull only what EDA needs (labels + images — skip val zip to save time)
mkdir -p data/raw/train_labels data/raw/train_images
aws s3 cp $BUCKET/xview/train_labels.zip data/raw/
aws s3 cp $BUCKET/xview/train_images.zip data/raw/

unzip -q data/raw/train_labels.zip -d data/raw/train_labels
unzip -q data/raw/train_images.zip  -d data/raw/train_images

# Start JupyterLab — no password, accessible on port 8888
cd /home/ec2-user/argus
jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --notebook-dir=/home/ec2-user/argus/notebooks \
    &>> /home/ec2-user/jupyter.log &

echo "JupyterLab started. Access at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8888"
