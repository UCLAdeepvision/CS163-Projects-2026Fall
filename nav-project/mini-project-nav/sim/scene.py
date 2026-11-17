"""One benchmark scene: what the robot sees, what it can hit, and where it may drive.

A scene folder in evaluation-scenes/ holds
    splat.ply   the 3D Gaussian splat that is rendered into the robot's camera
    mesh.ply    a triangle mesh of the same place, used only for collision checking
    grid.npz    a top-down map: ground height and a walkable mask on a regular grid

World frame: meters, z up. Yaw is measured counter-clockwise from +x. In the robot frame x is forward and y is left.
"""
import json
import os

import numpy as np

# Collision box of the robot (half extents in meters, x forward / y left / z up) and its height above the ground. It is the bounding box of the
# COCO robot (coco_one.usd: 0.76 m long, 0.61 m wide including the wheels, 0.71 m tall), centered on the robot's pose, from 0.15 m above the ground
# (so that the wheels do not touch the ground mesh) to the top of the robot.
ROBOT_HALF_EXTENTS = (0.38, 0.305, 0.279)
ROBOT_CLEARANCE = 0.15


class Scene:
    def __init__(self, root, scene_id, episodes):
        self.id = scene_id
        self.dir = os.path.join(root, "evaluation-scenes", scene_id)
        self.episodes = episodes                            # the benchmark entries for this scene

        for name in ("splat.ply", "mesh.ply", "grid.npz"):
            if not os.path.exists(os.path.join(self.dir, name)):
                raise FileNotFoundError(f"{os.path.join(self.dir, name)} is missing. Save nav-assets.tar in the nav-project folder and run `python unpack_files.py` in the mini-project-nav folder.")

        grid = np.load(os.path.join(self.dir, "grid.npz"))
        self.ground_z = grid["ground_z"]                    # ground height per cell
        self.walkable = grid["walkable"].astype(bool)       # may the robot be here
        self.origin = grid["origin_xy"].astype(np.float64)  # world position of the grid's corner
        self.res = float(grid["resolution_m"][0])           # cell size in meters

        # the collision mesh is loaded on first use: rendering alone does not need it
        self._collider = None
        self._box = None

    # ----- the map ------------------------------------------------------------------------------------------------------

    def _cell(self, x, y):
        """(row, column) of the grid cell containing the world point (x, y)."""
        return int(np.floor((y - self.origin[1]) / self.res)), int(np.floor((x - self.origin[0]) / self.res))

    def ground(self, x, y):
        """Ground height at (x, y). Points outside the map take the height of the nearest edge cell."""
        row, col = self._cell(x, y)
        row = min(max(row, 0), self.ground_z.shape[0] - 1)
        col = min(max(col, 0), self.ground_z.shape[1] - 1)
        return float(self.ground_z[row, col])

    def on_walkable(self, x, y):
        """True unless the robot is off the sidewalk, which means none of the 3x3 map cells around it is walkable."""
        row, col = self._cell(x, y)
        h, w = self.walkable.shape
        return bool(self.walkable[max(row - 1, 0):min(row + 2, h), max(col - 1, 0):min(col + 2, w)].any())

    # ----- camera and collisions ----------------------------------------------------------------------------------------

    def camera_pose(self, x, y, yaw, height, pitch=0.0):
        """Camera-to-world matrix (OpenCV axes: x right, y down, z forward) for a camera `height` meters above the ground."""
        forward = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        down = np.cross(forward, right)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2] = right, down, forward
        pose[:3, 3] = [x, y, self.ground(x, y) + height]
        return pose

    def in_collision(self, x, y, yaw):
        """True if the robot's box, standing at (x, y) with heading yaw, touches the mesh."""
        import trimesh                                      # imported here so that rendering works without it
        if self._collider is None:
            self._collider = trimesh.collision.CollisionManager()
            self._collider.add_object("scene", trimesh.load(os.path.join(self.dir, "mesh.ply"), process=False))
            self._box = trimesh.creation.box(extents=[2 * h for h in ROBOT_HALF_EXTENTS])
        pose = np.eye(4)
        c, s = np.cos(yaw), np.sin(yaw)
        pose[:2, :2] = [[c, -s], [s, c]]
        pose[:3, 3] = [x, y, self.ground(x, y) + ROBOT_CLEARANCE + ROBOT_HALF_EXTENTS[2]]
        return bool(self._collider.in_collision_single(self._box, transform=pose))


def load_benchmark(root, name="dev"):
    """Read the benchmark file evaluation-scenes/<name>.json.

    Returns the camera settings (the same for every scene) and a dict {scene id: Scene}.
    """
    with open(os.path.join(root, "evaluation-scenes", f"{name}.json")) as f:
        meta = json.load(f)
    scenes = {sid: Scene(root, sid, s["episodes"]) for sid, s in meta["scenes"].items()}
    return meta["camera"], scenes
