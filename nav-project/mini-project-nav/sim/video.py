"""First-person video of an episode with the policy's trajectory drawn on the camera image.

Like the videos of SidewalkBench-GS, each frame is the robot's camera view with the path on the ground drawn over it:
    green band   the trajectory the policy predicted (a policy exposes it as `self.waypoints`, [K, 2] meters, forward and left)
    orange line  the arc the commanded speed and yaw rate will drive, as far as the steering allows (next two seconds)
    white line   the route ahead (this is the benchmark's route; the policy itself never sees it)
    gold circle  the goal the policy was told
A SUCCESS or FAILURE banner is held for about a second at the end. One frame per policy step (4 Hz), written at real-time speed. The picture of the goal the policy was given is
inset in the top right corner.
"""
import math

import cv2
import imageio
import numpy as np

from .episode import MAX_STEER, WHEEL_BASE

GREEN, ORANGE, WHITE, GOLD = (60, 220, 90), (255, 150, 30), (255, 255, 255), (255, 210, 0)
ROBOT_HALF_WIDTH = 0.305        # m, half the width of the robot: the corridor drawn around the predicted trajectory
NEAR = 0.3                      # m: points closer than this to the camera plane are not drawn


def draw_path_corridor(img, waypoints, camera, half_width=0.28, color=(0, 0, 255)):
    """Draw the path a policy predicted on a camera picture, as the blue corridor of the SidewalkBench videos
    (sidewalkbench/pg/utils/vis.py, draw_trajectory_corridor). `waypoints` are [K, 2] meters in the robot frame (forward, left);
    the corridor is `half_width` meters to each side, projected onto flat ground, and fades with distance. `img` (RGB, any size
    with the camera's shape) is changed in place and returned. Only for display: the policy never sees it."""
    wp = np.asarray(waypoints, np.float32)[:, :2]
    if len(wp) < 2 or wp[-1, 0] <= 0.5:                                  # nothing to show for a path that does not lead forward
        return img
    if wp[0, 0] > 0.4:
        wp = np.vstack([[0.3, 0.0], wp]).astype(np.float32)             # start just in front of the robot
    step = np.diff(wp, axis=0)
    step = np.vstack([step, step[-1:]])
    length = np.linalg.norm(step, axis=1, keepdims=True)
    wp, step, length = wp[length[:, 0] > 1e-6], step[length[:, 0] > 1e-6], length[length[:, 0] > 1e-6]
    if len(wp) < 2:
        return img
    normal = np.stack([-step[:, 1], step[:, 0]], 1) / length

    h, w = img.shape[:2]
    focal = 0.5 * w / math.tan(math.radians(camera["fov"]) / 2)

    def pixels(points):                                                  # robot-frame ground points -> pixels (pinhole, flat ground)
        points = points[points[:, 0] > 1e-3]
        return np.stack([w / 2 - focal * points[:, 1] / points[:, 0], h / 2 + focal * camera["height_m"] / points[:, 0]], 1)

    left, right = pixels(wp + half_width * normal), pixels(wp - half_width * normal)
    if len(left) < 2 or len(right) < 2:
        return img
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.vstack([left, right[::-1]]).astype(np.int32)], 255)
    ys, xs = np.where(mask > 0)
    if len(ys):
        strength = 0.2 + 0.8 * (ys - ys.min()) / (ys.max() - ys.min() + 1e-6)       # stronger near the robot
        img[ys, xs] = (img[ys, xs] * (1 - strength[:, None]) + np.array(color, np.float32) * strength[:, None]).astype(np.uint8)
    return img


class EpisodeVideo:
    def __init__(self, path, scene, camera, policy, label="", scale=4, fps=4):
        self.scene, self.camera, self.policy, self.label, self.scale = scene, camera, policy, label, scale
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1, ffmpeg_log_level="error")
        self.fps, self.last = fps, None
        self.focal = 0.5 * camera["width"] / math.tan(math.radians(camera["fov"]) / 2)

    # ----- geometry -----------------------------------------------------------------------------------------------------

    def to_pixels(self, state, points):
        """World points [N, 3] -> (pixel positions [N, 2] in the enlarged image, mask of the points in front of the camera)."""
        c2w = self.scene.camera_pose(state.x, state.y, state.yaw, self.camera["height_m"])
        p = (points - c2w[:3, 3]) @ c2w[:3, :3]                         # camera coordinates: right, down, forward
        ok = p[:, 2] > NEAR
        z = np.where(ok, p[:, 2], 1.0)
        u = self.camera["width"] / 2 + self.focal * p[:, 0] / z
        v = self.camera["height"] / 2 + self.focal * p[:, 1] / z
        return np.stack([u, v], 1) * self.scale, ok

    def ground(self, xy):
        """Lift ground-plane points [N, 2] (world) onto the ground surface."""
        return np.array([[x, y, self.scene.ground(x, y)] for x, y in xy])

    def robot_to_world(self, state, points):
        """Robot-frame (forward, left) points [N, 2] -> world (x, y)."""
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        points = np.asarray(points, float)
        return np.stack([state.x + c * points[:, 0] - s * points[:, 1], state.y + s * points[:, 0] + c * points[:, 1]], 1)

    # ----- drawing ------------------------------------------------------------------------------------------------------

    def polyline(self, img, state, xy_world, color, thickness, dots=False):
        px, ok = self.to_pixels(state, self.ground(xy_world))
        for a in range(len(px) - 1):
            if ok[a] and ok[a + 1]:
                cv2.line(img, tuple(map(int, px[a])), tuple(map(int, px[a + 1])), color, thickness, cv2.LINE_AA)
        if dots:
            for a in range(len(px)):
                if ok[a]:
                    cv2.circle(img, tuple(map(int, px[a])), thickness + 2, color, -1, cv2.LINE_AA)

    def corridor(self, img, state, path):
        """Translucent band of the robot's width along `path` [N, 2] (robot frame, starting at the robot)."""
        overlay = img.copy()
        d = np.gradient(path, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        normal = np.stack([-d[:, 1], d[:, 0]], 1)
        left = self.robot_to_world(state, path + ROBOT_HALF_WIDTH * normal)
        right = self.robot_to_world(state, path - ROBOT_HALF_WIDTH * normal)
        pl, okl = self.to_pixels(state, self.ground(left))
        pr, okr = self.to_pixels(state, self.ground(right))
        for a in range(len(path) - 1):
            if okl[a] and okl[a + 1] and okr[a] and okr[a + 1]:
                quad = np.array([pl[a], pl[a + 1], pr[a + 1], pr[a]], np.int32)
                cv2.fillConvexPoly(overlay, quad, GREEN)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    def commanded_arc(self, episode):
        """Robot-frame points of the arc that the current command drives for two seconds, with the steering limit applied."""
        v, w = episode.command
        if abs(v) < 0.1:
            return None
        steer = float(np.clip(math.atan(w * WHEEL_BASE / v), -MAX_STEER, MAX_STEER))
        w = v * math.tan(steer) / WHEEL_BASE
        t = np.linspace(0, 2, 21)
        if abs(w) < 1e-4:
            return np.stack([v * t, np.zeros_like(t)], 1)
        return np.stack([v * np.sin(w * t) / w, v * (1 - np.cos(w * t)) / w], 1)

    def text(self, img, string, origin, scale=0.6, color=WHITE, thickness=1):
        (w, h), _ = cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = origin
        cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + 6), (0, 0, 0), -1)
        cv2.putText(img, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw(self, episode, image):
        s = episode.state
        img = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

        # the route ahead and the goal
        near = episode._nearest
        keep = (episode.cum >= near - 0.5) & (episode.cum <= near + 20.0)
        if keep.sum() >= 2:
            self.polyline(img, s, episode.line[keep], WHITE, 2)
        goal = episode.goal_point()
        px, ok = self.to_pixels(s, self.ground(np.array([goal])))
        if ok[0]:
            cv2.circle(img, tuple(map(int, px[0])), 4 * self.scale, GOLD, 3, cv2.LINE_AA)

        # what the policy predicted, and what it is driving
        waypoints = getattr(self.policy, "waypoints", None)
        if waypoints is not None:
            path = np.vstack([[0.0, 0.0], np.asarray(waypoints, float)[:, :2]])
            self.corridor(img, s, path)
            self.polyline(img, s, self.robot_to_world(s, path), GREEN, 2, dots=True)
        arc = self.commanded_arc(episode)
        if arc is not None:
            self.polyline(img, s, self.robot_to_world(s, arc), ORANGE, 3)

        # readout and legend
        self.text(img, f"{self.label}   t {episode.t:4.1f} s   v {s.v:.2f} m/s   steer {math.degrees(s.steer):+.0f} deg   "
                       f"goal {episode.goal_distance():.1f} m", (10, 24))
        for k, (name, color) in enumerate([("predicted trajectory", GREEN), ("commanded arc", ORANGE), ("route", WHITE), ("goal", GOLD)]):
            self.text(img, name, (10, img.shape[0] - 12 - 26 * k), 0.5, color)

        goal_image = getattr(episode, "goal_image", None)           # the picture of the end of the route, in a gold frame
        if goal_image is not None:
            w = img.shape[1] // 4
            small = cv2.resize(goal_image, (w, int(w * goal_image.shape[0] / goal_image.shape[1])), interpolation=cv2.INTER_AREA)
            img[8:8 + small.shape[0], img.shape[1] - 8 - w:img.shape[1] - 8] = small
            cv2.rectangle(img, (img.shape[1] - 8 - w, 8), (img.shape[1] - 9, 7 + small.shape[0]), GOLD, 2)
        return img

    # ----- writing ------------------------------------------------------------------------------------------------------

    def add(self, episode, image):
        """Use as the on_step callback of run_episode: one frame per policy step."""
        if image is None:
            return
        self.last = self.draw(episode, image)
        self.writer.append_data(self.last)

    def finish(self, result):
        """Hold the last frame under a SUCCESS or FAILURE banner for about a second, and close the file."""
        if self.last is not None:
            frame = self.last.copy()
            color = (40, 170, 60) if result["success"] else (200, 40, 40)
            band = frame.copy()
            cv2.rectangle(band, (0, frame.shape[0] // 2 - 50), (frame.shape[1], frame.shape[0] // 2 + 50), color, -1)
            cv2.addWeighted(band, 0.6, frame, 0.4, 0, frame)
            words = "SUCCESS" if result["success"] else f"FAILURE: {result['reason']}"
            (w, h), _ = cv2.getTextSize(words, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
            cv2.putText(frame, words, ((frame.shape[1] - w) // 2, frame.shape[0] // 2 + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 3, cv2.LINE_AA)
            for _ in range(self.fps):
                self.writer.append_data(frame)
        self.writer.close()
