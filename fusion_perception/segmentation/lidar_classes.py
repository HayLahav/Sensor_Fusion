"""LiDAR semantic classes used by the Utonia linear probe.

KITTI-360 label IDs (Cityscapes IDs 0–33 plus KITTI-360 extras 34–44) are
merged into 13 training classes. Points with IGNORE are not used in training
or evaluation.
"""
from __future__ import annotations
import numpy as np

IGNORE = 255

CLASS_NAMES = [
    "road",          # 0
    "sidewalk",      # 1
    "other-ground",  # 2
    "terrain",       # 3
    "building",      # 4
    "fence-wall",    # 5
    "pole-sign",     # 6
    "vegetation",    # 7
    "car",           # 8
    "truck-bus",     # 9
    "person",        # 10
    "two-wheeler",   # 11
    "other-object",  # 12
]
NUM_CLASSES = len(CLASS_NAMES)

# KITTI-360 label id → training class
_KITTI360_TO_TRAIN: dict[int, int] = {
    7: 0,                                   # road
    8: 1,                                   # sidewalk
    6: 2, 9: 2, 10: 2,                      # ground, parking, rail track
    22: 3,                                  # terrain
    11: 4, 15: 4, 16: 4, 34: 4,             # building, bridge, tunnel, garage
    12: 5, 13: 5, 14: 5, 35: 5,             # wall, fence, guard rail, gate
    17: 6, 18: 6, 19: 6, 20: 6, 37: 6, 38: 6,  # pole(group), traffic light/sign, smallpole, lamp
    21: 7,                                  # vegetation
    26: 8, 29: 8, 30: 8, 43: 8,             # car, caravan, trailer, unknown vehicle
    27: 9, 28: 9, 31: 9,                    # truck, bus, train
    24: 10, 25: 10,                         # person, rider
    32: 11, 33: 11,                         # motorcycle, bicycle
    4: 12, 5: 12, 36: 12, 39: 12, 40: 12, 41: 12, 42: 12, 44: 12,  # static/dynamic/other objects
}

# Detector class name → training classes that count as "the object's points"
DETECTION_CLASS_TO_LIDAR: dict[str, frozenset[int]] = {
    "car": frozenset({8, 9}),
    "vehicle": frozenset({8, 9}),
    "truck": frozenset({8, 9}),
    "bus": frozenset({8, 9}),
    "person": frozenset({10}),
    "pedestrian": frozenset({10}),
    "cyclist": frozenset({10, 11}),
    "bicycle": frozenset({10, 11}),
    "motorcycle": frozenset({10, 11}),
}


def kitti360_to_train(ids: np.ndarray) -> np.ndarray:
    """Map an array of KITTI-360 label ids to training classes (IGNORE if unmapped)."""
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for k, v in _KITTI360_TO_TRAIN.items():
        lut[k] = v
    ids = np.asarray(ids).astype(np.int64)
    out = np.full(ids.shape, IGNORE, dtype=np.uint8)
    ok = (ids >= 0) & (ids < 256)
    out[ok] = lut[ids[ok]]
    return out
