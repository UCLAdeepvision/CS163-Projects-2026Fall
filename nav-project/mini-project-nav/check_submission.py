"""Check an ONNX file before you submit it.

    python check_submission.py submission.onnx

Verifies that the file meets the interface in code/export_onnx.py and is safe to run, times one model call, and lets the model
drive for a few seconds in the simulator. A file that passes can still score badly, but it will run.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, "code")]

MAX_MB = 1000                   # largest file accepted (the original model is 97 MB)
MAX_FRAMES = 25                 # the simulator keeps this many past images
WAYPOINTS = 5                   # waypoints the model returns, 0.25 s apart
IMAGE_SHAPE = [126, 224, 3]     # the simulator's camera: height, width, channels
GOAL_SHAPE = [2]
INPUTS = {"images", "goal_image", "goal"}


def shape_of(value_info):
    """Shape of an ONNX input or output as a list, with None for dimensions that are not fixed."""
    return [d.dim_value if d.HasField("dim_value") else None for d in value_info.type.tensor_type.shape.dim]


def mismatch(shape, expected):
    """True if a recorded shape [N, ...] contradicts the expected one. Dimensions the file does not record are not held against it;
    run_problems checks the real shapes by running the model."""
    return len(shape) != len(expected) + 1 or any(s is not None and s != e for s, e in zip(shape[1:], expected))


def static_problems(path):
    """Problems visible in the file itself, without running it."""
    import onnx
    from onnx import TensorProto

    problems = []
    size = os.path.getsize(path) / 1e6
    if size > MAX_MB:
        problems.append(f"the file is {size:.0f} MB; the limit is {MAX_MB} MB")
    model = onnx.load(path, load_external_data=False)

    if any(t.data_location == TensorProto.EXTERNAL for t in model.graph.initializer):
        problems.append("the weights are stored in separate files; save everything in one .onnx file")
    domains = {op.domain for op in model.opset_import} | {node.domain for node in model.graph.node}
    if domains - {"", "ai.onnx"}:
        problems.append(f"custom operator domains are not allowed: {sorted(domains - {'', 'ai.onnx'})}")

    weights = {t.name for t in model.graph.initializer}
    inputs = {i.name: i for i in model.graph.input if i.name not in weights}       # older files list the weights as inputs too
    if set(inputs) != INPUTS:
        problems.append(f'the inputs must be named "images", "goal_image" and "goal", this file has {sorted(inputs)}')
    else:
        images, goal_image, goal = inputs["images"], inputs["goal_image"], inputs["goal"]
        shape = shape_of(images)
        if len(shape) != 5 or shape[2:] != IMAGE_SHAPE or shape[1] is None or not 1 <= shape[1] <= MAX_FRAMES or shape[0] not in (None, 1):
            problems.append(f'input "images" must have shape [1, T, 126, 224, 3] with T fixed between 1 and {MAX_FRAMES}, this file has {shape}')
        if images.type.tensor_type.elem_type != TensorProto.UINT8:
            problems.append('input "images" must be uint8')
        if mismatch(shape_of(goal_image), IMAGE_SHAPE) or shape_of(goal_image)[0] not in (None, 1):
            problems.append(f'input "goal_image" must have shape [1, 126, 224, 3], this file has {shape_of(goal_image)}')
        if goal_image.type.tensor_type.elem_type != TensorProto.UINT8:
            problems.append('input "goal_image" must be uint8')
        if mismatch(shape_of(goal), GOAL_SHAPE):
            problems.append(f'input "goal" must have shape [N, 2], this file has {shape_of(goal)}')
        if goal.type.tensor_type.elem_type != TensorProto.FLOAT:
            problems.append('input "goal" must be float32')
    outputs = [(o.name, shape_of(o)) for o in model.graph.output]
    if len(outputs) != 1 or outputs[0][0] != "waypoints":
        problems.append(f'there must be exactly one output, named "waypoints"; this file has {[o[0] for o in outputs]}')
    elif len(outputs[0][1]) != 3 or outputs[0][1][1] not in (None, WAYPOINTS) or outputs[0][1][2] not in (None, 2):
        problems.append(f'output "waypoints" must have shape [1, {WAYPOINTS}, 2], this file has {outputs[0][1]}')
    return problems


def run_problems(path):
    """Run the model on random inputs; returns (problems, seconds per call)."""
    import onnxruntime

    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    frames = {i.name: i for i in session.get_inputs()}["images"].shape[1]
    rng = np.random.default_rng(0)
    feed = {"images": rng.integers(0, 256, (1, frames, *IMAGE_SHAPE), dtype=np.uint8),
            "goal_image": rng.integers(0, 256, (1, *IMAGE_SHAPE), dtype=np.uint8), "goal": np.array([[6.0, 0.0]], np.float32)}
    out = session.run(["waypoints"], feed)[0]
    problems = []
    if out.shape != (1, WAYPOINTS, 2):
        problems.append(f"the output has shape {out.shape} for a batch of one, expected (1, {WAYPOINTS}, 2)")
    if not np.isfinite(out).all():
        problems.append("the output contains NaN or infinity")
    times = []
    for _ in range(10):
        start = time.time()
        session.run(["waypoints"], feed)
        times.append(time.time() - start)
    seconds = float(np.median(times))
    return problems, seconds


def drive(path, steps=10):
    """Let the model drive for a few seconds in the first scene, through the same policy the graders use."""
    from sim.episode import Episode, PHYS_HZ, POLICY_HZ
    from sim.runner import Renderer, load_policy, observe
    from sim.scene import load_benchmark

    os.environ["SUBMISSION_ONNX"] = os.path.abspath(path)
    camera, scenes = load_benchmark(ROOT)
    scene = scenes["8am"]
    episode = Episode(scene, scene.episodes[1])
    renderer, policy = Renderer(camera, "cpu"), load_policy(os.path.join(ROOT, "code", "onnx_policy.py"))
    goal_image = renderer.image(scene, *episode.goal_pose())
    frames, poses = [], []
    for _ in range(steps):
        v, w = policy.act(observe(renderer, scene, episode, frames, poses, goal_image))
        if not (np.isfinite(v) and np.isfinite(w)):
            raise RuntimeError("the policy returned a non-finite command")
        for _ in range(PHYS_HZ // POLICY_HZ):
            episode.step(v, w)
    return episode.path_len


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", help="the file to check")
    ap.add_argument("--no_drive", action="store_true", help="skip the short drive in the simulator")
    args = ap.parse_args()

    problems, seconds = [], None
    try:
        problems = static_problems(args.onnx)
        if not problems:                              # running a file that is malformed would only add noise
            problems, seconds = run_problems(args.onnx)
            if not problems and not args.no_drive:
                print(f"drive: the robot moved {drive(args.onnx):.1f} m in 2.5 s")
    except Exception as e:
        problems.append(f"{type(e).__name__}: {e}")
    if problems:
        print("\nThe file cannot be submitted:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"\nOK: {args.onnx} meets the interface ({seconds * 1000:.0f} ms per model call on this machine).")


if __name__ == "__main__":
    main()
