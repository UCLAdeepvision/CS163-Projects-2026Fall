"""Checks that the simulator and the model work on this machine.

    python check_setup.py

Runs six small tests (versions, packages, collision checking, rendering, the model, a short drive) and says which one failed.
"""
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, "code")]       # the simulator (sim/) and the model code (code/vint.py)

failed = []


def check(name, test):
    """Run test(); print its result (or its error) on the line that starts with `name`."""
    print(f"{name:<28}", end="", flush=True)
    try:
        print("ok", test() or "")
    except Exception:
        print("FAILED")
        traceback.print_exc()
        failed.append(name)


def versions():
    import numpy
    import torch
    import trimesh
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is needed")
    if torch.cuda.is_available():
        gpu = "cuda"
    elif torch.backends.mps.is_available():
        gpu = "mps available (not used by default)"
    else:
        gpu = "no GPU"
    return f"python {sys.version.split()[0]}, numpy {numpy.__version__}, torch {torch.__version__}, trimesh {trimesh.__version__}, {gpu}"


SCENES = ("8am", "dark", "hill", "round")


def files():
    """The big files do not come with the code: unpack_files.py unpacks them from nav-assets.tar. Say if any are missing and how to fix it."""
    missing = [p for p in [os.path.join("code", "vint.pt")] + [os.path.join("evaluation-scenes", s, f) for s in SCENES for f in ("splat.ply", "mesh.ply", "grid.npz")]
               if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise RuntimeError(f"{len(missing)} files are missing (for example {missing[0]}). Save nav-assets.tar in the nav-project folder and run `python unpack_files.py` in this folder.")


def packages():
    for name in ("cv2", "fcl", "imageio_ffmpeg", "efficientnet_pytorch", "onnxruntime", "PIL", "plyfile", "scipy", "viser"):
        __import__(name)


def collision():
    """A box next to a wall must collide, and the same box three meters away must not."""
    import numpy as np
    import trimesh
    manager = trimesh.collision.CollisionManager()
    manager.add_object("wall", trimesh.creation.box(extents=[4, 0.2, 2]))
    box = trimesh.creation.box(extents=[0.44, 0.30, 0.44])
    near, far = np.eye(4), np.eye(4)
    far[:3, 3] = [0, 3, 0]
    assert manager.in_collision_single(box, transform=near) and not manager.in_collision_single(box, transform=far)


def rendering():
    from sim.runner import Renderer
    from sim.scene import load_benchmark
    camera, scenes = load_benchmark(ROOT)
    scene = scenes["8am"]
    spec = scene.episodes[0]
    start = time.time()
    image = Renderer(camera, "cpu").image(scene, spec["start"][0], spec["start"][1], spec["yaw"])
    if image.mean() < 5:
        raise RuntimeError("the rendered image is black")
    return f"{image.shape[1]}x{image.shape[0]} frame, {time.time() - start:.1f} s including loading the scene"


def model():
    import torch
    from vint import forward, load_model
    net = load_model().eval()
    with torch.no_grad():
        out = forward(net, torch.zeros(1, 18, 64, 85), torch.zeros(1, 3, 64, 85))
    assert tuple(out.shape) == (1, 5, 2)


def drive():
    """Let the policy drive for ten steps (2 s) through the whole loop: render, observe, act, physics."""
    from sim.episode import Episode, PHYS_HZ, POLICY_HZ
    from sim.runner import Renderer, load_policy, observe
    from sim.scene import load_benchmark
    camera, scenes = load_benchmark(ROOT)
    scene = scenes["8am"]
    episode = Episode(scene, scene.episodes[1])
    renderer = Renderer(camera, "cpu")
    policy = load_policy(os.path.join(ROOT, "code", "policy.py"))
    goal_image = renderer.image(scene, *episode.goal_pose())
    frames, poses = [], []
    for _ in range(10):
        v, w = policy.act(observe(renderer, scene, episode, frames, poses, goal_image))
        for _ in range(PHYS_HZ // POLICY_HZ):           # one policy step is ten physics steps
            episode.step(v, w)
    return f"10 policy steps (2.5 s), robot moved {episode.path_len:.1f} m"


check("files", files)
check("versions", versions)
check("packages", packages)
check("collision checking", collision)
check("rendering (CPU)", rendering)
check("model", model)
check("short drive", drive)
print("\nEverything is fine." if not failed else f"\nFailed: {', '.join(failed)}. See the messages above and the troubleshooting table in the handout.")
sys.exit(1 if failed else 0)
