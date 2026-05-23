# Fusion Perception Pipeline

A real-time autonomous driving perception pipeline that fuses monocular camera and LiDAR inputs using open-vocabulary 3D detection, multi-object tracking, BEV occupancy mapping, and language-based scene reasoning.

## Architecture

```
Video / KITTI sequence
        │
        ▼
 WildDet3D (3D detection, open-vocabulary text prompts)
        │  Detection3D list per frame
        ▼
 KalmanCoWTracker (multi-object tracking)
        │  Track list with 3D centroid history
        ▼
 BEV Occupancy Grid (LiDAR fusion + track centroids)
        │  80×100 confidence grid
        ▼
 Gemma 4 2B (language scene reasoning, every N frames)
        │  Natural language summary + recommended action
        ▼
 Output: annotated video · detections.json · tracks.json · reasoning.json
```

## Key Components

| Component | File | Description |
|-----------|------|-------------|
| 3D Detector | `fusion_perception/models/wilddet3d_wrapper.py` | Wraps [WildDet3D](https://github.com/allenai/WildDet3D) — open-vocabulary monocular 3D detection with text prompts |
| Tracker | `fusion_perception/tracking/kalman_cowtracker.py` | Kalman filter + [CoWTracker](https://github.com/facebookresearch/co-tracker) visual tracking; ByteTrack two-stage association |
| BEV Grid | `fusion_perception/occupancy/bev_grid.py` | Bird's-eye-view occupancy: exponential decay → LiDAR rasterization (0.6) → track centroids (1.0) |
| Reasoner | `fusion_perception/models/gemma_wrapper.py` | Gemma 4 2B-IT with 4-bit quantization; text-only or visual mode |
| Prompt Builder | `fusion_perception/reasoning/prompt_builder.py` | Converts `SceneMemory` to structured scene description |
| py123d Loader | `fusion_perception/data/py123d_loader.py` | Streams KITTI-360 / Waymo frames via [py123d](https://github.com/py123d/py123d); real calibration + LiDAR |
| Evaluation | `fusion_perception/evaluation/` | mAP · MOTA/MOTP · occupancy IoU; cross-dataset benchmark report |

## Quickstart (local)

```bash
# 1. Install
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/allenai/WildDet3D
pip install git+https://github.com/facebookresearch/co-tracker
pip install -e .

# 2. Download WildDet3D checkpoint
mkdir -p ckpt
huggingface-cli download allenai/WildDet3D \
    wilddet3d_alldata_all_prompt_v1.0.pt --local-dir ckpt/

# 3. Run on a video
python run.py --video path/to/clip.mp4

# Merge extra configs
python run.py --config configs/default.yaml configs/colab.yaml --video clip.mp4
```

Outputs are written to `outputs/<run_id>/`:
- `output.mp4` — annotated frame | BEV grid | reasoning panel
- `detections.json` — per-frame 3D detections
- `tracks.json` — track history with 3D centroid trajectories
- `reasoning.json` — Gemma reasoning outputs (includes `prompt_used` + `summary`)

## Google Colab (T4 GPU)

Open `notebooks/pipeline_finetune.ipynb` in Colab with a **T4 GPU** runtime.

The notebook runs the full pipeline end-to-end on the KITTI `2011_10_03_drive_0047` sequence, then fine-tunes Gemma 4 2B on the collected reasoning outputs — all in one place.

**Sections:**
1. Mount Drive, install all dependencies
2. Download KITTI data, parse real camera intrinsics
3. Run pipeline (prints each reasoning output live)
4. Inspect training data — statistics, length histogram, example pairs
5. QLoRA fine-tuning with loss curve visualization
6. Evaluate fine-tuned adapter vs. base model

Alternatively, use the standalone notebooks:
- `notebooks/00_setup.ipynb` — environment setup + pipeline run only
- `notebooks/01_finetune_gemma.ipynb` — fine-tuning only (loads from existing `reasoning.json`)

## Configuration

All config lives in `configs/`. Configs are merged left-to-right:

```yaml
# configs/default.yaml (key settings)
detection:
  prompts: ["car", "person", "cyclist"]   # open-vocabulary
  score_threshold: 0.4

tracking:
  backend: "kalman_cow"   # "kalman_cow" | "cowtracker"
  ego_motion: true        # ORB homography compensation

occupancy:
  resolution: 0.5         # meters per cell
  z_range: [0.0, 50.0]   # 50 m forward range
  lidar_confidence: 0.6   # confidence for LiDAR-only cells

reasoning:
  enabled: true
  interval_frames: 30     # run Gemma every 30 frames
  quantize_4bit: true
```

## Fine-tuning Gemma 4 2B

The pipeline saves `prompt_used` and `summary` in every `reasoning.json`. These become training pairs automatically:

```python
# Each reasoning.json entry becomes:
{
  "messages": [
    {"role": "user",      "content": "<scene prompt>"},
    {"role": "assistant", "content": "<gemma summary>"}
  ]
}
```

Fine-tuning uses standard HuggingFace `peft` + `trl` (QLoRA, no extra libraries):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("google/gemma-4-2b-it", ...)
model = PeftModel.from_pretrained(base, "ckpt/gemma4_lora")
```

See `notebooks/pipeline_finetune.ipynb` (Section 4) for the full training setup.

## Benchmarking

```python
from omegaconf import OmegaConf
from fusion_perception.evaluation import BenchmarkRunner

cfg = OmegaConf.merge(
    OmegaConf.load("configs/default.yaml"),
    OmegaConf.load("configs/benchmark.yaml"),
)
runner = BenchmarkRunner(cfg, datasets=["kitti-360", "waymo"], logs_per_dataset=3)
report = runner.run()
# Writes outputs/benchmark/<date>/report.md + report.json
```

Metrics reported per dataset and per log:
- **mAP** — mean average precision at IoU 0.5 (BEV axis-aligned boxes)
- **MOTA** — multi-object tracking accuracy (CLEAR MOT)
- **MOTP** — multi-object tracking precision (mean BEV center distance)
- **Occ IoU** — binary occupancy grid IoU vs. ground-truth LiDAR

Requires [py123d](https://github.com/py123d/py123d) and `PY123D_DATA_ROOT` pointing to your datasets directory.

## Project Structure

```
fusion_perception/
├── data/           # py123d dataset loader
├── evaluation/     # metrics, benchmark evaluator, report generator
├── models/         # WildDet3D wrapper, Gemma wrapper
├── occupancy/      # BEV grid with LiDAR fusion
├── pipelines/      # StreamingPipeline, StageRunner
├── reasoning/      # prompt builder, templates
├── tracking/       # KalmanCoWTracker, association, ego-motion
├── utils/          # dataclasses, geometry, logging, video loader
└── visualization/  # BEV renderer, frame annotator, compositor

configs/            # YAML configs (default, colab, benchmark, ...)
notebooks/          # Colab notebooks
tests/              # pytest unit tests (~20 test files)
run.py              # CLI entrypoint
```

## Requirements

- Python 3.10+
- CUDA GPU with ≥10 GB VRAM (T4 or better)
- See `requirements.txt` for Python packages

## License

MIT
