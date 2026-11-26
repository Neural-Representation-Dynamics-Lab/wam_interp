"""Geometry and measurements for a paired, open-loop obstacle experiment (metres)."""

from dataclasses import asdict, dataclass
from xml.etree import ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class Barrier:
    center: tuple
    forward: tuple
    half_sizes: tuple
    side: int
    off_route: bool

    @property
    def lateral(self):
        return np.array([-self.forward[1], self.forward[0]])

    def xml(self, xml):
        tree = ET.fromstring(xml)
        world = tree.find("worldbody")
        if world is None or tree.find(".//geom[@name='e1_barrier']") is not None:
            raise ValueError("Expected one world and no existing E1 barrier")
        yaw = np.arctan2(self.forward[1], self.forward[0])
        fmt = lambda seq: " ".join(f"{v:.12g}" for v in seq)
        # A jointless world body keeps LIBERO's flattened state layout intact.
        body = ET.SubElement(
            world,
            "body",
            name="e1_barrier_body",
            pos=fmt(self.center),
            quat=fmt([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
        )
        ET.SubElement(
            body,
            "geom",
            name="e1_barrier",
            type="box",
            size=fmt(self.half_sizes),
            group="1",
            rgba=".12 .63 .86 1",
            contype="1",
            conaffinity="1",
        )
        return ET.tostring(tree, encoding="unicode")

    def clearance(self, path):
        """Signed distance from the gripper reference point to the oriented solid."""
        vec = np.asarray(path) - self.center
        local = np.stack(
            (vec[..., :2] @ self.forward, vec[..., :2] @ self.lateral, vec[..., 2]),
            axis=-1,
        )
        d = np.abs(local) - self.half_sizes
        return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(axis=-1), 0)

    def as_dict(self):
        return asdict(self)


def placement(path, side, *, off_route=False, off_distance=0.14, forward_offset=0.0):
    path = np.asarray(path, dtype=float)
    if path.shape != (33, 3) or not np.isfinite(path).all() or side not in (-1, 1):
        raise ValueError("Expected a finite 32-control reach and a signed side")
    heading = path[-1, :2] - path[0, :2]
    distance = np.linalg.norm(heading)
    if distance < 0.025:
        raise ValueError("Less than 25 mm of planar reference travel")
    forward = heading / distance
    lateral = np.array([-forward[1], forward[0]])
    # A world-fixed horizontal bar at the late reach, above tabletop clutter.
    xy = path[28, :2] + side * (off_distance if off_route else 0.024) * lateral
    xy = xy + forward_offset * forward
    return Barrier(
        tuple(xy) + (path[28, 2],),
        tuple(forward),
        (0.012, 0.037, 0.012),
        side,
        off_route,
    )


def score(path, baseline, barrier, step):
    path, baseline = np.asarray(path), np.asarray(baseline)
    if not 0 < step < min(len(path), len(baseline)):
        raise ValueError("Readout exceeds paired trajectory")
    delta = (path[step] - path[0]) - (baseline[step] - baseline[0])
    return {
        "open_side_mm": float(-barrier.side * 1000 * (delta[:2] @ barrier.lateral)),
        "forward_mm": float(1000 * (delta[:2] @ barrier.forward)),
        "min_eef_clearance_mm": float(1000 * barrier.clearance(path[: step + 1]).min()),
    }
