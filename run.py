"""CLI entrypoint for the fusion perception pipeline.

Usage:
    python run.py --config configs/default.yaml
    python run.py --config configs/default.yaml --video path/to/clip.mp4
    python run.py --config configs/default.yaml configs/colab.yaml  # merge configs
"""
import argparse
from omegaconf import OmegaConf
from fusion_perception.pipelines.streaming_pipeline import StreamingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fusion Perception Pipeline")
    parser.add_argument("--config", nargs="+", default=["configs/default.yaml"],
                        help="One or more YAML config files (merged left-to-right)")
    parser.add_argument("--video", default=None,
                        help="Override video source path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.merge(*[OmegaConf.load(f) for f in args.config])
    if args.video:
        cfg.video.source = args.video

    pipeline = StreamingPipeline(cfg)
    pipeline.run(cfg.video.source)


if __name__ == "__main__":
    main()
