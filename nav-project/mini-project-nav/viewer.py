"""Watch a policy drive in the browser (built with viser).

    python viewer.py --policy code/policy.py --checkpoint runs/mine/last.pt

then open http://localhost:8080 and keep that tab visible. Your browser draws the 3D scene and also the robot's camera
pictures (what the policy sees), which is fast on any computer with a graphics card. Those pictures differ a little from the
ones evaluate.py renders, so an episode can end differently here: the scores of evaluate.py are the ones that count.
With --sim_camera the camera pictures come from the simulator's own renderer instead, and "Run all" gives the same numbers as
evaluate.py (slow without an NVIDIA GPU).
"""
import argparse
import os
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import viser
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from sim.episode import summarize
from sim.runner import Renderer, load_policy, run_episode
from sim.video import EpisodeVideo, draw_path_corridor
from sim.scene import load_benchmark
from sim.splat import covariance, read_ply

ROOT = os.path.dirname(os.path.abspath(__file__))
ROBOT_URDF = os.path.join(ROOT, "assets", "coco_one", "coco_one.urdf")      # the COCO robot (from github.com/VAIL-UCLA/coco2play)
WHEEL_RADIUS = 0.11                 # m
ROBOT_ORIGIN_HEIGHT = 0.277         # m: the URDF's origin is this far above the lowest point of the wheels

ROUTE_COLOR = (40, 110, 255)
GOAL_COLOR = (40, 220, 90)
TRACE_COLOR = (255, 60, 60)
HIT_COLOR = (255, 230, 0)
DRAW_EVERY = 4                      # redraw the robot every fourth physics step (10 times per second)


def splat_arrays(path):
    """The arrays viser's add_gaussian_splats wants: centers, covariances, colors, opacities."""
    g = read_ply(path)
    cov = covariance(torch.from_numpy(g["quat"]), torch.from_numpy(g["scale"])).numpy()
    # viser sends covariances as float16 (largest value 65504, i.e. a 256 m standard deviation). A few distant sky Gaussians are
    # larger than that and would become infinite, so shrink just those to fit.
    largest = np.abs(cov).max(axis=(1, 2))
    cov *= np.minimum(1.0, 65000.0 / np.maximum(largest, 1e-12))[:, None, None]
    return g["mean"], cov, g["color"], g["opacity"][:, None]


def yaw_quaternion(yaw):
    """Rotation about the vertical axis as the (w, x, y, z) quaternion viser expects."""
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


class BrowserCamera:
    """The robot's camera pictures, drawn by your browser's GPU (viser). This is the viewer's default; --sim_camera uses
    sim/splat.py instead.

    The pictures look slightly different from the ones evaluate.py renders (another splat renderer), so the scores of evaluate.py
    are the ones that count. This class has the same `camera` and `image(scene, x, y, yaw)` as sim.runner.Renderer.
    """

    def __init__(self, server, camera, set_overlays):
        self.server, self.camera, self.set_overlays = server, camera, set_overlays
        self.lock = threading.Lock()        # one picture at a time: taking one moves the browser's camera
        self.ready = None                   # (browser tab, scene) whose splats are known to be loaded in the browser
        self.first_person = False           # during an episode the browser's view stays at the robot's eye (no switching back and forth, which flickers)

    def client(self):
        clients = list(self.server.get_clients().values())
        if not clients:
            raise RuntimeError("open the viewer in a browser first: the browser draws the robot's camera pictures (or start with --sim_camera)")
        return clients[0]

    def image(self, scene, x, y, yaw):
        """uint8 [H, W, 3] view from the robot standing at (x, y) with heading yaw, as the browser draws it."""
        cam = self.camera
        width, height = cam["width"], cam["height"]
        client = self.client()
        pose = scene.camera_pose(x, y, yaw, cam["height_m"])
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()                       # x, y, z, w
        canvas_w, canvas_h = client.camera.image_width, client.camera.image_height      # the browser's drawing area
        if canvas_w < width or canvas_h < height:
            raise RuntimeError(f"the browser window is too small ({canvas_w} x {canvas_h} pixels): make it larger")
        canvas, target = canvas_w / canvas_h, width / height
        # The browser's vertical field of view is chosen so that, once its picture is cut to the shape of the policy's picture
        # (width : height), the horizontal field of view is exactly cam["fov"].
        tan_half = np.tan(np.radians(cam["fov"]) / 2) / (target if canvas >= target else canvas)

        def shot():
            h = min(2 * height, canvas_h)
            try:
                image = client.get_render(h, int(round(h * canvas)), transport_format="jpeg", timeout=30)[..., :3]
            except TimeoutError:
                raise RuntimeError("the browser did not send a camera picture: keep the viewer's tab open and visible") from None
            h, w = image.shape[:2]
            if w / h > target:                                                 # cut the middle part that has the shape of the policy's picture
                cut = int(round(h * target)); image = image[:, (w - cut) // 2:(w - cut) // 2 + cut]
            else:
                cut = int(round(w / target)); image = image[(h - cut) // 2:(h - cut) // 2 + cut]
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        with self.lock:
            switch = not self.first_person                                     # outside an episode: go to the robot's eye for one picture, then back
            before = (client.camera.position, client.camera.wxyz, client.camera.fov)
            with client.atomic():
                client.camera.position = tuple(float(v) for v in pose[:3, 3])
                client.camera.wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
                client.camera.fov = float(2 * np.arctan(tan_half))
            if switch:
                self.set_overlays(False)                                       # the route, the goal and the robot are not in the policy's picture
            try:
                image = shot()
                if self.ready != (client.client_id, scene.id):                 # a new tab or a new scene: wait until the browser has loaded the splats,
                    for _ in range(60):                                        # that is, until the picture is not blank and has stopped changing
                        time.sleep(0.5)
                        last, image = image, shot()
                        if image.std() > 4 and np.abs(image.astype(int) - last.astype(int)).mean() < 1.0:
                            break
                    self.ready = (client.client_id, scene.id)
            finally:
                if switch:
                    self.set_overlays(True)
                    with client.atomic():
                        client.camera.position, client.camera.wxyz, client.camera.fov = before
        return image

    def look_from(self, scene, x, y, yaw):
        """Put the browser's camera at the robot's eye (first-person mode), so the view moves smoothly between the policy's pictures."""
        clients = list(self.server.get_clients().values())
        if clients:
            pose = scene.camera_pose(x, y, yaw, self.camera["height_m"])
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            with clients[0].atomic():
                clients[0].camera.position = tuple(float(v) for v in pose[:3, 3])
                clients[0].camera.wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


class Viewer:
    def __init__(self, args):
        self.args = args
        self.camera, self.scenes = load_benchmark(args.root, args.benchmark)
        self.renderer = Renderer(self.camera, args.device, keep=1)

        self.worker = None                  # the thread that runs episodes, so the browser stays responsive
        self.stop = threading.Event()       # set by the Stop button
        self.results = []                   # results of the episodes finished in the current run
        self.shown_scene = None             # the scene whose splats are in the browser
        self.hits = []                      # markers of the collisions drawn so far
        self.route_handle = self.goal_handle = self.trace_handle = None     # what is drawn on top of the scene (hidden when the browser takes the policy's picture)
        self.robot = None                   # the robot's frame in the browser: its mesh is sent once, then only moved
        self.trace = []                     # points of the robot's path in the current episode
        self.steps = 0                      # physics steps taken in the current episode
        self.policy = None                  # the policy of the running episode (its predicted path is drawn on the camera picture)
        self.view_lock = threading.RLock()  # show() is called from several threads (menus, a new browser tab, the episode worker)

        self.server = viser.ViserServer(host=args.host, port=args.port, verbose=False)
        self.server.scene.set_up_direction("+z")
        if not args.sim_camera:
            self.renderer = BrowserCamera(self.server, self.camera, self.set_overlays)
        self.build_panel()
        self.show(list(self.scenes)[0], 0)

        @self.server.on_client_connect
        def _(client):                      # a tab was opened: point its camera at the robot and show the robot's camera picture
            if self.worker is None or not self.worker.is_alive():
                threading.Thread(target=lambda: self.show(self.scene_menu.value, self.route_index()), daemon=True).start()

    # ----- the control panel --------------------------------------------------------------------------------------------

    def route_labels(self, scene_id):
        return [f"{i}: {e['length']:.0f} m, turn {e['turn_deg']:.0f} deg" for i, e in enumerate(self.scenes[scene_id].episodes)]

    def build_panel(self):
        gui = self.server.gui
        scene_id = list(self.scenes)[0]
        self.scene_menu = gui.add_dropdown("Scene", list(self.scenes), initial_value=scene_id)
        self.route_menu = gui.add_dropdown("Route", self.route_labels(scene_id), initial_value=self.route_labels(scene_id)[0])
        self.speed_menu = gui.add_dropdown("Speed", ["real time", "max"], initial_value="max")
        self.follow = gui.add_checkbox("Follow the robot", True)
        self.run_one = gui.add_button("Run episode", color="blue")
        self.run_all = gui.add_button("Run all")
        self.stop_button = gui.add_button("Stop", color="red", disabled=True)
        self.status = gui.add_text("Status", "ready", disabled=True)
        self.clock = gui.add_text("Time / distance to goal", "-", disabled=True)
        self.image = gui.add_image(np.zeros((self.camera["height"], self.camera["width"], 3), np.uint8), label="Policy camera", format="jpeg")
        self.table = gui.add_markdown("")

        self.scene_menu.on_update(lambda _: self.on_scene_changed())
        self.route_menu.on_update(lambda _: self.show(self.scene_menu.value, self.route_index()))
        self.run_one.on_click(lambda _: self.start(all_episodes=False))
        self.run_all.on_click(lambda _: self.start(all_episodes=True))
        self.stop_button.on_click(lambda _: self.stop.set())

    def route_index(self):
        return self.route_labels(self.scene_menu.value).index(self.route_menu.value)

    def on_scene_changed(self):
        labels = self.route_labels(self.scene_menu.value)
        self.route_menu.options = labels
        self.route_menu.value = labels[0]
        self.show(self.scene_menu.value, 0)

    def set_controls_enabled(self, enabled):
        for widget in (self.run_one, self.run_all, self.scene_menu, self.route_menu):
            widget.disabled = not enabled
        self.stop_button.disabled = enabled

    # ----- the 3D view --------------------------------------------------------------------------------------------------

    def show(self, scene_id, index):
        """Show a scene with one route: the route, the goal, the robot at its start and the camera view from there."""
        with self.view_lock:
            scene = self.scenes[scene_id]
            view = self.server.scene
            if self.shown_scene != scene_id:                    # loading the splats takes a moment, so only do it on a change
                view.add_gaussian_splats("/splat", *splat_arrays(os.path.join(scene.dir, "splat.ply")))
                self.shown_scene = scene_id
            spec = scene.episodes[index]
            route = np.array(spec["centerline"])
            z = scene.ground(*route[0]) + 0.08
            self.route_handle = view.add_spline_catmull_rom("/route", np.c_[route, np.full(len(route), z)][::2], color=ROUTE_COLOR, thickness=4.0, thickness_units="screen")
            self.goal_handle = view.add_icosphere("/goal", radius=0.5, color=GOAL_COLOR, opacity=0.55,
                               position=(route[-1][0], route[-1][1], scene.ground(*route[-1]) + 0.5))
            self.last_pose = None
            self.place_robot(scene, *spec["start"], spec["yaw"])
            self.trace_handle = view.add_point_cloud("/trace", np.array([[route[0][0], route[0][1], z]], np.float32), TRACE_COLOR, point_size=0.06, point_shape="circle")
            for marker in self.hits:
                marker.remove()
            self.hits = []
            self.move_camera(scene, *spec["start"], spec["yaw"], back=4.0, up=3.0)
            if not (isinstance(self.renderer, BrowserCamera) and not self.server.get_clients()):      # the browser draws the picture, so one has to be open
                self.image.image = self.renderer.image(scene, spec["start"][0], spec["start"][1], spec["yaw"])

    def place_robot(self, scene, x, y, yaw, steer=0.0):
        """Put the robot at (x, y) with heading yaw and front-axle steering angle `steer`; the wheels roll by the distance driven since the last call."""
        view = self.server.scene
        if self.robot is None:                              # send the robot once, then only move its frame and turn its joints
            frame = view.add_frame("/robot", show_axes=False)
            self.urdf = ViserUrdf(self.server, urdf_or_path=Path(ROBOT_URDF), root_node_name="/robot/urdf")
            self.wheel_angle, self.last_pose = 0.0, None
            self.robot = frame                              # set last: the robot counts as made only when all of it is
        if self.last_pose is not None:
            self.wheel_angle += ((x - self.last_pose[0]) * np.cos(yaw) + (y - self.last_pose[1]) * np.sin(yaw)) / WHEEL_RADIUS
        self.last_pose = (x, y)
        self.urdf.update_cfg(np.array([steer, self.wheel_angle, self.wheel_angle, self.wheel_angle, self.wheel_angle]))     # also positions the axle and the wheels
        self.robot.position = (x, y, scene.ground(x, y) + ROBOT_ORIGIN_HEIGHT)
        self.robot.wxyz = yaw_quaternion(yaw)

    def set_overlays(self, visible):
        """Show or hide the route, the goal, the trace, the collision markers and the robot."""
        for handle in (self.route_handle, self.goal_handle, self.trace_handle, self.robot, *self.hits):
            if handle is not None:
                handle.visible = visible

    def move_camera(self, scene, x, y, yaw, back=3.0, up=2.0):
        """Put every browser's camera `back` meters behind and `up` meters above the robot, looking along its heading."""
        z = scene.ground(x, y)
        for client in self.server.get_clients().values():
            client.camera.position = (x - back * np.cos(yaw), y - back * np.sin(yaw), z + up)
            client.camera.look_at = (x + 2 * np.cos(yaw), y + 2 * np.sin(yaw), z + 0.3)

    # ----- running episodes ---------------------------------------------------------------------------------------------

    def start(self, all_episodes):
        if self.worker is not None and self.worker.is_alive():
            return
        if all_episodes:
            jobs = [(sid, i) for sid, s in self.scenes.items() for i in range(len(s.episodes))]
        else:
            jobs = [(self.scene_menu.value, self.route_index())]
        self.results = []
        self.stop.clear()
        self.set_controls_enabled(False)
        self.worker = threading.Thread(target=self.run, args=(jobs,), daemon=True)
        self.worker.start()

    def run(self, jobs):
        """Worker thread: run the episodes in `jobs`, a list of (scene id, route index)."""
        realtime = 1.0 if self.speed_menu.value == "real time" else 0.0
        try:
            for scene_id, index in jobs:
                if self.stop.is_set():
                    break
                self.show(scene_id, index)
                scene, spec = self.scenes[scene_id], self.scenes[scene_id].episodes[index]
                policy = load_policy(self.args.policy)                      # a fresh policy for every episode
                self.policy = policy
                self.status.value = f"running {scene_id} route {index}"
                self.trace = [np.array([*spec["start"], scene.ground(*spec["start"]) + 0.1])]
                self.steps = 0
                video = None
                if self.args.save_video:
                    os.makedirs(self.args.save_video, exist_ok=True)
                    video = EpisodeVideo(os.path.join(self.args.save_video, f"{scene_id}_{index}.mp4"), scene, self.camera, policy,
                                         f"{scene_id} {index}")

                def on_step(episode, image):
                    self.on_step(scene, episode, image)
                    if video:
                        video.add(episode, image)

                first_person = isinstance(self.renderer, BrowserCamera)       # the browser draws the camera pictures: its view stays at the
                if first_person:                                               # robot's eye for the episode, with the robot and the route hidden
                    fov = [c.camera.fov for c in self.server.get_clients().values()]
                    self.set_overlays(False)
                    self.renderer.first_person = True
                try:
                    result, episode = run_episode(self.renderer, scene, spec, policy, should_stop=self.stop.is_set, realtime=realtime,
                                                  on_step=on_step)
                finally:
                    if first_person:
                        self.renderer.first_person = False
                        self.set_overlays(True)
                        for c, f in zip(self.server.get_clients().values(), fov):
                            c.camera.fov = f
                if first_person:
                    self.move_camera(scene, episode.state.x, episode.state.y, episode.state.yaw)      # back to the view from behind the robot
                if video:
                    video.finish(result)
                if result["reason"] != "stopped":
                    self.results.append(result)
                    self.table.content = self.summary()
                self.status.value = f"{'reached the goal' if result['success'] else 'failed'}: {result['reason']}"
        except Exception:
            traceback.print_exc()
            self.status.value = "error, see the terminal"
        finally:
            self.set_controls_enabled(True)

    def on_step(self, scene, episode, image):
        """Called after every physics step: update the camera picture, the robot, its path and the collision markers."""
        if image is not None:
            self.image.image = self.camera_picture(image)
        self.steps += 1
        if self.steps % DRAW_EVERY and not episode.done:
            return
        s = episode.state
        first_person = isinstance(self.renderer, BrowserCamera) and self.renderer.first_person
        self.place_robot(scene, s.x, s.y, s.yaw, s.steer)
        self.trace.append(np.array([s.x, s.y, scene.ground(s.x, s.y) + 0.1]))
        if len(self.trace) % 4 == 0 or episode.done:
            self.trace_handle = self.server.scene.add_point_cloud("/trace", np.array(self.trace, np.float32), TRACE_COLOR, point_size=0.07, point_shape="circle")
            self.trace_handle.visible = not first_person
        while len(self.hits) < len(episode.collision_points):
            hx, hy = episode.collision_points[len(self.hits)]
            self.hits.append(self.server.scene.add_icosphere(f"/hit{len(self.hits)}", radius=0.25, color=HIT_COLOR,
                                                              position=(hx, hy, scene.ground(hx, hy) + 0.3)))
        self.clock.value = f"{episode.t:5.1f} s / {episode.goal_distance():5.1f} m"
        if first_person:
            self.renderer.look_from(scene, s.x, s.y, s.yaw)
        elif self.follow.value:
            self.move_camera(scene, s.x, s.y, s.yaw)

    def camera_picture(self, image):
        """The picture for the "Policy camera" panel: the policy's image, with the path it predicted drawn as a blue corridor
        (as in the SidewalkBench videos). The policy itself gets the picture without it."""
        waypoints = getattr(self.policy, "waypoints", None)
        if self.args.no_path_overlay or waypoints is None:
            return image
        return draw_path_corridor(cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR), waypoints, self.camera)

    def summary(self):
        s = summarize(self.results)
        return (f"**Score {s['score']:.1f}** (reached the goal without a collision), route completion {100 * s['route_completion']:.0f}%, "
                f"{100 * s['collision_rate']:.0f}% of the episodes ended in a collision ({s['episodes']} episodes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="python file that defines class Policy")
    ap.add_argument("--checkpoint", default=None, help="weights for the policy; it receives the path in the CHECKPOINT environment variable")
    ap.add_argument("--onnx", default=None, help="a model to score with code/onnx_policy.py (the policy receives the path in the SUBMISSION_ONNX environment variable)")
    ap.add_argument("--benchmark", default="dev", help="name of the episode list in evaluation-scenes/")
    ap.add_argument("--root", default=ROOT, help="folder that contains evaluation-scenes/")
    ap.add_argument("--device", default="auto", help="where to render: auto (CUDA if present, else CPU), cpu, cuda or mps")
    ap.add_argument("--sim_camera", action="store_true", help="render the robot's camera pictures with the simulator's own renderer (sim/splat.py), exactly as evaluate.py does, instead of taking them from the browser: slow without an NVIDIA GPU")
    ap.add_argument("--no_path_overlay", action="store_true", help="do not draw the path the policy predicted (the blue corridor) on the camera picture")
    ap.add_argument("--save_video", default=None, help="folder to write a first-person video of every episode, with the policy's trajectory drawn on it")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="localhost")
    args = ap.parse_args()

    if args.checkpoint:
        os.environ["CHECKPOINT"] = os.path.abspath(args.checkpoint)
    if args.onnx:
        os.environ["SUBMISSION_ONNX"] = os.path.abspath(args.onnx)
    Viewer(args)
    print(f"open http://{args.host}:{args.port} in your browser (Ctrl+C to quit)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
