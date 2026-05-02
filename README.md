# ARGUS Targeting System

AI-powered satellite object detection system that identifies military targets — tanks, vehicles, naval vessels, aircraft, and structures — from overhead satellite imagery and outputs GPS coordinates.

---

## Baseline Model

The first trained model is a **YOLOv8s** detector trained on the xView satellite dataset, remapped from 60 raw classes to 5 ARGUS target classes. Training was done on AWS EC2 using a two-phase frozen-backbone strategy.

### Results

| Metric | Value |
|--------|-------|
| Model | YOLOv8s |
| mAP50 | 0.459 |
| mAP50-95 | — |
| Training Platform | AWS EC2 g4dn.xlarge (Tesla T4 GPU) |
| Training Time | ~4 hours |
| Dataset | xView (satellite imagery) |
| Image Size | 640×640 |

### Classes

| ID | Class | xView Source IDs | Train Instances |
|----|-------|-----------------|-----------------|
| 0 | tank | 58 | 318,698 |
| 1 | large_vehicle | 17, 21, 23, 53 | 28,864 |
| 2 | naval_vessel | 40, 41, 42 | 215,849 |
| 3 | aircraft | 11, 12 | 14,053 |
| 4 | structure | 60, 61, 62 | 3,199 |

### Known Limitations

- **Naval vessel false positives** — the model fires on ocean background due to 215,849 vessel training instances dominating gradients
- **Small object misses** — aircraft and structure (under 20px at 640px chip) frequently not detected
- **Class imbalance** — 100× ratio between most and least frequent class (tank vs structure) hurts rare class recall
- **No geographic filtering** — vessel detections on land are not suppressed
- **Single model ceiling** — all five classes compete in the same detection head, cross-class confusion cannot be structurally eliminated

---

## Training Pipeline

### Phase 1 — Frozen Backbone (50 epochs)

The backbone (CSPDarknet, first 10 layers) is frozen. Only the FPN neck and detection head are trained. This prevents destroying ImageNet/COCO pretrained features on early epochs when the head weights are still random.

```
lr0    = 0.01
freeze = 10
epochs = 50
batch  = 8
```

### Phase 2 — Full Fine-Tune (50 epochs)

Loads Phase 1 best weights, unfreezes all layers, continues training with a lower learning rate for final refinement.

```
lr0    = 0.001
freeze = 0
epochs = 50
batch  = 8
patience = 10
```

### Training Infrastructure

| Component | Value |
|-----------|-------|
| EC2 Instance | g4dn.xlarge |
| GPU | NVIDIA Tesla T4 (16GB VRAM) |
| AMI | Deep Learning Base AMI (Amazon Linux 2023) |
| Root Volume | 30GB (OS + CUDA drivers) |
| Data Volume | 125GB EBS (packages + dataset + outputs) |
| IAM Role | ec2-s3-access |
| S3 Bucket | s3://argus-training-data-890615325560-us-east-1-an |

The EC2 instance auto-terminates after training completes. All weights, plots, and metrics are uploaded to S3.

---

## Repository Structure

```
Argus_Targeting_System/
├── configs/
│   └── dataset.yaml              # xView 5-class dataset config
├── docs/
│   └── argus_phase_plan.md       # Full multi-agent architecture plan
├── scripts/
│   ├── ec2_train.sh              # EC2 bootstrap + training launch script
│   ├── ec2_preprocess.sh         # EC2 preprocessing script
│   ├── ec2_eda.sh                # EC2 EDA script
│   ├── batch_inference.py        # Run inference on full image directory
│   ├── inference_test.py         # Single image inference test
│   └── test_local.py             # Local sanity checks
├── src/
│   ├── training/
│   │   └── train.py              # Two-phase YOLOv8 training script
│   ├── inference/                # Inference pipeline
│   ├── preprocessing/            # xView chipping + label conversion
│   ├── ingestion/                # Data ingestion utilities
│   ├── api/                      # FastAPI serving layer
│   └── database/                 # PostGIS database models
├── models/
│   └── argus-v1/                 # Trained model weights (S3 synced)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install Dependencies

```bash
git clone https://github.com/Adnan082/Argus_Target_System.git
cd Argus_Target_System
python -m venv argus-env
source argus-env/bin/activate      # Windows: argus-env\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Inference on a Single Image

```bash
python scripts/inference_test.py \
    --image path/to/image.jpg \
    --model models/argus-v1/phase2/weights/best.pt \
    --conf 0.25 \
    --save
```

Output:
```
Image: path/to/image.jpg
Detections: 3
  tank             conf=0.81  box=[120,200,180,260]
  naval_vessel     conf=0.74  box=[300,400,420,480]
  large_vehicle    conf=0.61  box=[50,50,90,90]
```

### 3. Run Inference on a Directory

```bash
python scripts/batch_inference.py \
    --dir data/processed/images/val/ \
    --model models/argus-v1/phase2/weights/best.pt \
    --conf 0.25 \
    --save
```

Output:
```
========================================
Images processed:        1200
Images with detections:  843
Total detections:        4271
Avg detections/image:    3.6

Detections by class:
  naval_vessel     2104
  tank             1022
  large_vehicle    688
  aircraft         312
  structure        145
```

### 4. Reproduce Training on AWS EC2

```bash
# Launch EC2 g4dn.xlarge with IAM role ec2-s3-access
# Use as User Data script:
bash scripts/ec2_train.sh
```

The script:
1. Mounts the 125GB EBS data volume
2. Creates a Python virtualenv on the data volume
3. Pulls GitHub PAT from AWS Secrets Manager
4. Clones the repo
5. Syncs the processed dataset from S3
6. Runs two-phase training
7. Uploads all weights and metrics to S3
8. Self-terminates the EC2 instance

---

## Dataset

**xView** — WorldView-3 satellite imagery at 0.3m ground resolution, covering 1,415 km² across 60 object classes.

- Download: [xviewdataset.org](http://xviewdataset.org)
- License: CC BY-NC-SA 4.0
- Raw format: GeoTIFF with GeoJSON labels

The preprocessing pipeline chips each GeoTIFF into 640×640 tiles, converts labels from GeoJSON to YOLO format, and remaps the 60 xView classes to 5 ARGUS classes.

---

## Next Steps — Multi-Agent Architecture

The baseline single-model approach has a structural ceiling at mAP50 ≈ 0.459. The planned next stage replaces it with **5 specialist binary agents** — one per class — each trained to answer a single question: "is this object my target class, or background?"

| Agent | Model | Chip Size | Key Advantage |
|-------|-------|-----------|---------------|
| Tank | YOLOv8m | 640px | Dedicated backbone, no cross-class competition |
| Large Vehicle | YOLOv8m | 640px | Dedicated backbone |
| Naval Vessel | YOLOv8l | 1280px | Geographic water mask filter |
| Aircraft | YOLOv8m-p2 | 320px | P2 detection head for tiny objects |
| Structure | YOLOv8m-p2 | 320px | P2 detection head for tiny objects |

Each agent is trained in three phases:
- **Phase 0** — Synthetic pre-training on cut-paste renders (50 epochs)
- **Phase 1** — Frozen backbone fine-tune on xView (30 epochs)
- **Phase 2** — Full unfreeze fine-tune (40 epochs)

A supervisor agent merges detections via global NMS, converts pixel coordinates to GPS lat/lon, and applies scene-level reasoning. A PPO-trained RL agent then learns optimal chip scanning strategies.

Full architecture plan: [docs/argus_phase_plan.md](docs/argus_phase_plan.md)

---

## Requirements

- Python 3.10+
- CUDA 12.x (for GPU training)
- AWS account with S3 and EC2 access
- xView dataset license

---

## License

MIT License — see [LICENSE](LICENSE)
