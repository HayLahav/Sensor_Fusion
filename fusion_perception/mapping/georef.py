"""Place the KITTI-360 world frame (poses.txt, LiDAR map) on the globe.

KITTI-360 poses are metric but in a local world frame. KITTI-360 also ships
the raw GPS/IMU log (OXTS: latitude, longitude, altitude per frame). Fitting
one rigid transform (rotation + translation, optional scale) between the pose
positions and the GPS positions converted to a local East-North-Up frame
gives latitude / longitude for every map point, the route and any detection
or planned path.

The fit uses many frames and drops outliers, so single GPS glitches don't
matter, and its residual says how well the two agree (expect decimetres to
about a metre: GPS error, not pose error, dominates).
"""
from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass
from glob import glob
from typing import Optional
import numpy as np

_A = 6378137.0                 # WGS84
_E2 = 6.69437999014e-3


def lla_to_ecef(lat, lon, alt) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.stack([(n + alt) * np.cos(lat) * np.cos(lon),
                     (n + alt) * np.cos(lat) * np.sin(lon),
                     (n * (1 - _E2) + alt) * np.sin(lat)], -1)


def _enu_rotation(lat0, lon0) -> np.ndarray:
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array([[-np.sin(lo), np.cos(lo), 0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def lla_to_enu(lat, lon, alt, lat0, lon0, alt0) -> np.ndarray:
    d = lla_to_ecef(lat, lon, alt) - lla_to_ecef(lat0, lon0, alt0)
    return d @ _enu_rotation(lat0, lon0).T


def enu_to_lla(enu: np.ndarray, lat0, lon0, alt0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ecef = np.asarray(enu) @ _enu_rotation(lat0, lon0) + lla_to_ecef(lat0, lon0, alt0)
    x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - _E2))
    for _ in range(5):                         # Bowring-style iteration, converges fast
        n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - _E2 * n / (n + alt)))
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


def find_oxts_dir(root: str, sequence: str) -> Optional[str]:
    """Locate oxts/data for a drive in the usual KITTI-360 layouts."""
    for pat in (f"{root}/data_poses/{sequence}/oxts/data",
                f"{root}/data_poses_oxts/data_poses/{sequence}/oxts/data",
                f"{root}/data_poses_oxts/{sequence}/oxts/data",
                f"{root}/*/{sequence}/oxts/data"):
        hits = sorted(glob(pat))
        if hits:
            return hits[0]
    return None


def load_oxts(oxts_dir: str) -> dict[int, tuple[float, float, float]]:
    """{frame: (lat, lon, alt)} from one OXTS text file per frame."""
    out = {}
    for p in sorted(glob(os.path.join(oxts_dir, "*.txt"))):
        try:
            vals = open(p).read().split()
            out[int(os.path.splitext(os.path.basename(p))[0])] = (
                float(vals[0]), float(vals[1]), float(vals[2]))
        except (ValueError, IndexError):
            continue
    return out


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Least-squares R, t, s with dst ≈ s·R·src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


@dataclass
class GeoReference:
    lat0: float
    lon0: float
    alt0: float
    R: list                  # world → ENU rotation (3×3)
    t: list
    scale: float
    fitted_scale: float      # scale estimate even when not applied (≈1 means metric agreement)
    rms_horizontal_m: float
    rms_vertical_m: float
    frames_used: int
    frames_rejected: int

    def world_to_enu(self, pts: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(pts) @ np.asarray(self.R).T + np.asarray(self.t)

    def world_to_lla(self, pts: np.ndarray):
        return enu_to_lla(self.world_to_enu(pts), self.lat0, self.lon0, self.alt0)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GeoReference":
        with open(path) as f:
            return cls(**json.load(f))


def fit_georeference(
    world_xyz: dict[int, np.ndarray],
    gps_lla: dict[int, tuple[float, float, float]],
    with_scale: bool = False,
    reject_factor: float = 3.0,
    iterations: int = 3,
) -> GeoReference:
    """Fit world → ENU from frames that have both a pose and a GPS fix."""
    frames = sorted(set(world_xyz) & set(gps_lla))
    if len(frames) < 10:
        raise ValueError(f"only {len(frames)} frames have both pose and GPS; need ≥ 10")
    lla = np.array([gps_lla[f] for f in frames])
    lat0, lon0, alt0 = lla[0]
    enu = lla_to_enu(lla[:, 0], lla[:, 1], lla[:, 2], lat0, lon0, alt0)
    world = np.array([world_xyz[f] for f in frames], dtype=np.float64)

    keep = np.ones(len(frames), bool)
    for _ in range(iterations):
        R, t, s = umeyama(world[keep], enu[keep], with_scale)
        res = np.linalg.norm((s * world @ R.T + t) - enu, axis=1)
        new_keep = res <= reject_factor * max(np.median(res[keep]), 0.05)
        if (new_keep == keep).all():
            break
        keep = new_keep
    _, _, s_est = umeyama(world[keep], enu[keep], True)
    err = (s * world[keep] @ R.T + t) - enu[keep]
    return GeoReference(
        lat0=float(lat0), lon0=float(lon0), alt0=float(alt0),
        R=R.tolist(), t=t.tolist(), scale=float(s), fitted_scale=float(s_est),
        rms_horizontal_m=float(np.sqrt((err[:, :2] ** 2).sum(1).mean())),
        rms_vertical_m=float(np.sqrt((err[:, 2] ** 2).mean())),
        frames_used=int(keep.sum()), frames_rejected=int((~keep).sum()),
    )


# ── Exports ──────────────────────────────────────────────────────────────────

def lines_geojson(lines: list[dict]) -> dict:
    """lines: [{"lla": (lat, lon, alt arrays), "properties": {...}}] → GeoJSON."""
    feats = []
    for ln in lines:
        lat, lon, _ = ln["lla"]
        feats.append({"type": "Feature",
                      "geometry": {"type": "LineString",
                                   "coordinates": [[float(a), float(b)] for a, b in zip(lon, lat)]},
                      "properties": ln.get("properties", {})})
    return {"type": "FeatureCollection", "features": feats}


_LEAFLET_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiDAR Map View</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}#info{{position:absolute;z-index:1000;bottom:12px;left:12px;
background:rgba(255,255,255,.9);padding:6px 10px;font:13px system-ui;border-radius:6px}}</style>
</head><body><div id="map"></div><div id="info">{info}</div>
<script>
const map = L.map('map');
const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom: 21, maxNativeZoom: 19, attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
const lidar = L.imageOverlay('{image}', {bounds}, {{opacity: 0.75}}).addTo(map);
const lines = L.geoJSON({geojson}, {{style: f => ({{color: f.properties.color || '#00c853',
  weight: f.properties.weight || 3, opacity: 0.9}})}}).addTo(map);
L.control.layers(null, {{'LiDAR map': lidar, 'Paths': lines}}).addTo(map);
L.control.scale().addTo(map);
map.fitBounds({bounds});
</script></body></html>
"""


def write_leaflet_html(path: str, image_name: str, bounds_latlon: list, geojson: dict, info: str) -> None:
    """Map page: OpenStreetMap tiles, the north-up LiDAR raster and GeoJSON paths."""
    html = _LEAFLET_HTML.format(image=image_name, bounds=json.dumps(bounds_latlon),
                                geojson=json.dumps(geojson), info=info)
    with open(path, "w") as f:
        f.write(html)
