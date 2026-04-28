#!/bin/bash
# EC2 bootstrap: pull processed dataset from S3, train YOLOv8, push model to S3
# Launch with: Deep Learning Base AMI with Single CUDA (Amazon Linux 2023), g4dn.xlarge, IAM role ec2-s3-access
# Storage: Volume 1 (root, 30GB) = OS + CUDA drivers | Volume 2 (125GB) = packages + dataset + outputs
exec > /tmp/train.log 2>&1
set -e
echo "=== ARGUS Training started: $(date) ==="

BUCKET="s3://argus-training-data-890615325560-us-east-1-an"
REGION="us-east-1"

# Capture instance ID
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

# ── Step 1: Mount second EBS (250GB) BEFORE doing anything else ──────────────
echo "=== Mounting second EBS volume ==="
DATA_DEV=""
for dev in /dev/nvme1n1 /dev/nvme2n1 /dev/xvdb /dev/sdb; do
    if [ -b "$dev" ] && ! mount | grep -q "^$dev"; then
        DATA_DEV=$dev
        break
    fi
done

if [ -z "$DATA_DEV" ]; then
    echo "ERROR: Could not find second EBS volume. Exiting."
    exit 1
fi

mkfs.ext4 -F $DATA_DEV
mkdir -p /mnt/data
mount $DATA_DEV /mnt/data
echo "Mounted $DATA_DEV at /mnt/data — $(df -h /mnt/data | tail -1)"

# ── Step 2: Create Python virtualenv on second EBS ───────────────────────────
echo "=== Creating virtualenv on /mnt/data ==="
python3 -m venv /mnt/data/venv
source /mnt/data/venv/bin/activate

# All pip installs now go to /mnt/data/venv — root volume untouched
export TMPDIR=/mnt/data/tmp
export YOLO_CONFIG_DIR=/mnt/data/tmp/ultralytics
mkdir -p /mnt/data/tmp

pip install ultralytics boto3 --quiet

# ── Step 3: GitHub PAT from Secrets Manager ──────────────────────────────────
GITHUB_TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id argus/github-pat \
    --region $REGION \
    --query SecretString \
    --output text | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

# ── Step 4: Clone repo onto second EBS ───────────────────────────────────────
git clone https://Adnan082:${GITHUB_TOKEN}@github.com/Adnan082/Argus_Target_System.git /mnt/data/argus
cd /mnt/data/argus

# ── Step 5: Pull processed dataset from S3 onto second EBS ───────────────────
echo "=== Downloading processed dataset: $(date) ==="
mkdir -p /mnt/data/argus/data/processed
aws s3 sync $BUCKET/processed/ /mnt/data/argus/data/processed/ \
    --exclude "yolo_labels_raw/*"

# Fix absolute path in dataset.yaml
DATASET_YAML="/mnt/data/argus/data/processed/dataset.yaml"
sed -i "s|^path:.*|path: /mnt/data/argus/data/processed|" $DATASET_YAML
echo "=== dataset.yaml path after fix ==="
grep "^path:" $DATASET_YAML

# ── Step 6: Train — outputs go to /mnt/data/argus/runs automatically ─────────
echo "=== Running training: $(date) ==="
python src/training/train.py \
    --data $DATASET_YAML \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 16 \
    --workers 4 \
    --device 0 \
    --name argus-v1 \
    --s3-bucket $BUCKET

echo "=== Training complete: $(date) ==="

# Self-terminate
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
