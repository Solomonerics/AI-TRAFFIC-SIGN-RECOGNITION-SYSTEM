# Ghana Traffic Sign Recognition Prototype

PyTorch, Ultralytics YOLO, torchvision, and OpenCV prototype for detecting and recognizing traffic signs in images, video files, or a live camera stream.

The system uses two distinct models:

1. **Detector:** YOLO finds traffic sign bounding boxes in the full frame.
2. **Classifier:** MobileNetV3 classifies each cropped sign image.

The runtime pipeline is:

```text
camera/video/image frame
  -> YOLO detector finds traffic-sign boxes
  -> each detected box is cropped
  -> classifier predicts the sign class
  -> OpenCV draws the box, label, and confidence
```

This is currently a public-data prototype. The default dataset is `AlessandroFerrante/StreetSignSet` from Hugging Face. It is useful for bootstrapping detection/classification, but it is not a certified Ghana-only road-sign dataset. For a Ghana production model, collect Ghana road footage, label Ghana signs, and fine-tune from this baseline.

## Current Model Outputs

Default trained model paths:

```text
models/detector/weights/best.pt
models/classifier/best.pt
```

Current evaluation report path:

```text
reports/evaluation.json
```

Latest verified evaluation on the prepared public dataset:

| Component | Metric | Value |
| --- | ---: | ---: |
| Detector | mAP50 | 0.9477 |
| Detector | mAP50-95 | 0.7996 |
| Detector | precision | 0.9144 |
| Detector | recall | 0.9049 |
| Classifier | top-1 accuracy | 0.9500 |
| Classifier | test images | 260 |

## Requirements

Local development:

- Python project managed by `uv`
- NVIDIA GPU recommended for training
- Tested machine: RTX 3060 Laptop GPU, 6 GB VRAM
- Driver-reported CUDA: 12.7
- PyTorch wheel target: CUDA 12.6

Docker development:

- Docker Desktop or Docker Engine
- Docker Compose
- NVIDIA GPU support if training/evaluating on GPU
- On Windows, GPU containers usually require WSL2-backed Docker Desktop with NVIDIA GPU support enabled

## Project Layout

```text
configs/
  project.yaml          # default paths and training hyperparameters
  ghana_signs.yaml      # initial Ghana class list
  public_signs.yaml     # generated active public dataset class list

data/
  detector.yaml         # generated YOLO dataset config
  public/               # downloaded Hugging Face snapshot
  detector/             # prepared YOLO detector dataset
  classifier/           # prepared classifier crop dataset

models/
  detector/weights/     # YOLO checkpoints
  classifier/           # classifier checkpoints and metrics

reports/
  evaluation.json       # evaluation report output

traffic_sign_detector/
  cli.py                # command entry points
  detector.py           # YOLO training/evaluation
  classifier.py         # torchvision classifier training/loading
  public_data.py        # public dataset download/preparation
  realtime.py           # OpenCV realtime inference
  dataset.py            # validation and frame extraction helpers
```

Generated datasets, model weights, logs, and reports are intentionally ignored by Git.

## Local Setup

Install dependencies:

```powershell
uv sync --extra dev
```

Check the environment and GPU:

```powershell
uv run tsr-env
```

Expected GPU-ready signs:

- `cuda_available` is `true`
- GPU name includes `NVIDIA GeForce RTX 3060 Laptop GPU`

## Logging

Every CLI command supports:

```powershell
--log-level INFO --log-file logs/training.log
```

Monitor logs in another PowerShell terminal:

```powershell
Get-Content .\logs\training.log -Wait -Tail 80
```

Use a separate evaluation log when validating models:

```powershell
--log-file logs/evaluation.log
```

## Dataset Workflow

Download and prepare the default public dataset:

```powershell
uv run download-public-data --log-level INFO --log-file logs/data.log
```

This creates:

```text
data/public/street-sign-set/
data/detector/
data/classifier/
data/detector.yaml
configs/public_signs.yaml
```

If the Hugging Face download was already completed, prepare only:

```powershell
uv run prepare-public-data --log-level INFO --log-file logs/data.log
```

If the raw dataset changed and you want to regenerate prepared outputs:

```powershell
uv run prepare-public-data --clean --log-level INFO --log-file logs/data.log
```

Validate prepared detector and classifier data:

```powershell
uv run validate-data --strict --log-level INFO --log-file logs/data-validation.log
```

The detector data is converted to one class:

```text
traffic_sign
```

The classifier data keeps the active traffic-sign classes found in the usable public dataset and writes them to:

```text
configs/public_signs.yaml
```

## Training

Use the low-memory defaults for the RTX 3060 Laptop GPU.

Train the detector:

```powershell
uv run train-detector --epochs 50 --imgsz 640 --batch 4 --workers 0 --mosaic 0 --close-mosaic 50 --device 0 --log-level INFO --log-file logs/training.log
```

Resume detector training from the last checkpoint:

```powershell
uv run train-detector --resume-from models/detector/weights/last.pt --epochs 50 --imgsz 640 --batch 4 --workers 0 --mosaic 0 --close-mosaic 50 --device 0 --log-level INFO --log-file logs/training.log
```

Train the classifier:

```powershell
uv run train-classifier --epochs 25 --batch 16 --workers 0 --device cuda --log-level INFO --log-file logs/training.log
```

Classifier training uses ImageNet-pretrained MobileNetV3 by default. Horizontal flips are not used because left/right sign direction can change the label meaning.

## Testing

Run unit tests:

```powershell
uv run pytest
```

Run data validation as an integration check:

```powershell
uv run validate-data --strict --log-level INFO --log-file logs/data-validation.log
```

Run an environment smoke test:

```powershell
uv run tsr-env
```

## Evaluation Reports

Run detector and classifier evaluation on GPU:

```powershell
uv run evaluate --detector-data-yaml data/detector.yaml --classifier-data-root data/classifier --detector-batch 1 --detector-workers 0 --device 0 --output reports/evaluation.json --log-level INFO --log-file logs/evaluation.log
```

Stable CPU evaluation fallback:

```powershell
uv run evaluate --detector-data-yaml data/detector.yaml --classifier-data-root data/classifier --detector-batch 1 --detector-workers 0 --device cpu --output reports/evaluation.json --log-level INFO --log-file logs/evaluation.log
```

Print a compact report summary:

```powershell
uv run python -c "import json; from pathlib import Path; data=json.loads(Path('reports/evaluation.json').read_text()); print(json.dumps({'detector': data.get('detector'), 'classifier_top1_accuracy': data.get('classifier', {}).get('top1_accuracy'), 'classifier_total_images': data.get('classifier', {}).get('total_images')}, indent=2))"
```

The full `reports/evaluation.json` includes:

- detector `mAP50`
- detector `mAP50-95`
- detector precision
- detector recall
- classifier top-1 accuracy
- classifier confusion matrix

## Realtime Recognition

Run on the laptop webcam:

```powershell
uv run realtime --source 0 --device 0 --log-level INFO --log-file logs/realtime.log
```

Run on a video file:

```powershell
uv run realtime --source .\videos\sample.mp4 --device 0 --log-level INFO --log-file logs/realtime.log
```

Save an annotated video:

```powershell
uv run realtime --source .\videos\sample.mp4 --device 0 --output reports\annotated-demo.mp4 --log-level INFO --log-file logs/realtime.log
```

Run without opening a display window:

```powershell
uv run realtime --source .\videos\sample.mp4 --device 0 --no-display --output reports\annotated-demo.mp4
```

Low-confidence classifier predictions are shown as:

```text
unknown
```

## Docker

The Docker image includes the trained detector and classifier checkpoints, so the default container command is the live realtime demo:

```text
realtime --source 0 --device 0 --log-level INFO --log-file logs/realtime.log
```

Build and start the webcam demo:

```powershell
docker compose up --build
```

The bundled model files are:

```text
models/detector/weights/best.pt
models/classifier/best.pt
```

Compose mounts `data`, `logs`, `reports`, and `configs` from the host. It uses a Docker named volume for `models` so the trained checkpoints baked into the image are available immediately when the volume is first created.

Build the image without starting the demo:

```powershell
docker compose build
```

Check environment inside the container:

```powershell
docker compose run --rm app tsr-env
```

Download and prepare data:

```powershell
docker compose run --rm app download-public-data --log-level INFO --log-file logs/data.log
```

Validate data:

```powershell
docker compose run --rm app validate-data --strict --log-level INFO --log-file logs/data-validation.log
```

Train detector with GPU:

```powershell
docker compose run --rm app train-detector --epochs 50 --imgsz 640 --batch 4 --workers 0 --mosaic 0 --close-mosaic 50 --device 0 --log-level INFO --log-file logs/training.log
```

Resume detector training:

```powershell
docker compose run --rm app train-detector --resume-from models/detector/weights/last.pt --epochs 50 --imgsz 640 --batch 4 --workers 0 --mosaic 0 --close-mosaic 50 --device 0 --log-level INFO --log-file logs/training.log
```

Train classifier with GPU:

```powershell
docker compose run --rm app train-classifier --epochs 25 --batch 16 --workers 0 --device cuda --log-level INFO --log-file logs/training.log
```

Run tests:

```powershell
docker compose run --rm app pytest
```

Run evaluation:

```powershell
docker compose run --rm app evaluate --detector-data-yaml data/detector.yaml --classifier-data-root data/classifier --detector-batch 1 --detector-workers 0 --device 0 --output reports/evaluation.json --log-level INFO --log-file logs/evaluation.log
```

Run CPU evaluation if GPU passthrough is not available:

```powershell
docker compose run --rm app evaluate --detector-data-yaml data/detector.yaml --classifier-data-root data/classifier --detector-batch 1 --detector-workers 0 --device cpu --output reports/evaluation.json --log-level INFO --log-file logs/evaluation.log
```

Run realtime inference on a mounted video file:

```powershell
docker compose run --rm app realtime --source data/videos/sample.mp4 --device 0 --no-display --output reports/annotated-demo.mp4
```

Reset the Docker model volume if you rebuild the image with newer bundled checkpoints:

```powershell
docker compose down -v
docker compose up --build
```

Webcam and GUI-window access from Docker is host-dependent. The image defaults to the live webcam demo, but Docker Desktop on Windows may not expose the webcam/display directly to Linux containers. If webcam display fails in Docker on Windows, run the same demo locally:

```powershell
uv run realtime --source 0 --device 0 --log-level INFO --log-file logs/realtime.log
```

## Direct Docker Commands

Build without Compose:

```powershell
docker build -t traffic-sign-detector .
```

Run the environment check:

```powershell
docker run --rm --gpus all traffic-sign-detector tsr-env
```

Run the default realtime command:

```powershell
docker run --rm --gpus all traffic-sign-detector
```

Use `--device cpu` in project commands when running without `--gpus all`.

## CLI Reference

| Command | Purpose |
| --- | --- |
| `tsr-env` | Print Python, PyTorch, CUDA, and GPU information |
| `download-public-data` | Download and prepare the Hugging Face dataset |
| `prepare-public-data` | Prepare an existing raw dataset snapshot |
| `validate-data` | Validate YOLO labels, classifier folders, and class consistency |
| `train-detector` | Fine-tune YOLO detector checkpoints |
| `train-classifier` | Fine-tune the crop classifier |
| `evaluate` | Write detector/classifier/FPS metrics to JSON |
| `realtime` | Run OpenCV detection and recognition on webcam/video |
| `extract-frames` | Extract low-frequency video frames for later labeling |

## Troubleshooting

**CUDA validation fails with memory or unknown CUDA errors**

Use lower detector evaluation settings:

```powershell
--detector-batch 1 --detector-workers 0
```

If CUDA still fails, run evaluation on CPU:

```powershell
--device cpu
```

**Docker cannot see the GPU**

Check:

```powershell
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

If that fails, fix Docker/NVIDIA GPU passthrough before running GPU training in the project container.

**Hugging Face download is interrupted or rate-limited**

If enough files were downloaded, run:

```powershell
uv run prepare-public-data
```

The preparation step uses valid image/label pairs and writes an active class list for the available data.

**Classifier or detector paths are missing**

Run training first, or pass explicit checkpoint paths:

```powershell
--detector-model models/detector/weights/best.pt
--classifier-model models/classifier/best.pt
```

**OpenCV display fails in Docker**

Use `--no-display --output reports/annotated-demo.mp4`, or run realtime locally outside Docker.
