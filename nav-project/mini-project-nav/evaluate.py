"""Run a policy on the benchmark and write the results.

    python evaluate.py --policy code/policy.py --checkpoint runs/mine/last.pt
    python evaluate.py --policy code/onnx_policy.py --onnx submission.onnx   # score an exported model, as the graders do
    python evaluate.py --policy code/policy.py --scenes triangle loop        # only some scenes, while developing

Prints one line per episode and the summary, and writes the results to results/<policy name>.json.
"""
import argparse
import json
import os
import time

from sim.episode import summarize
from sim.video import EpisodeVideo
from sim.runner import Renderer, load_policy, run_episode
from sim.scene import load_benchmark

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="python file that defines class Policy")
    ap.add_argument("--checkpoint", default=None, help="weights for the policy; it receives the path in the CHECKPOINT environment variable")
    ap.add_argument("--onnx", default=None, help="a model to score with code/onnx_policy.py (the policy receives the path in the SUBMISSION_ONNX environment variable)")
    ap.add_argument("--benchmark", default="dev", help="name of the episode list in evaluation-scenes/")
    ap.add_argument("--root", default=ROOT, help="folder that contains evaluation-scenes/")
    ap.add_argument("--scenes", nargs="*", default=None, help="run only these scenes")
    ap.add_argument("--device", default="auto", help="where to render: auto (CUDA if present, else CPU), cpu, cuda or mps")
    ap.add_argument("--out", default=None, help="results file (default: results/<policy name>.json)")
    ap.add_argument("--save_video", default=None, help="folder to write one first-person video per episode, with the policy's trajectory drawn on it")
    args = ap.parse_args()

    if args.checkpoint:
        os.environ["CHECKPOINT"] = os.path.abspath(args.checkpoint)
    if args.onnx:
        os.environ["SUBMISSION_ONNX"] = os.path.abspath(args.onnx)
    camera, scenes = load_benchmark(args.root, args.benchmark)
    renderer = Renderer(camera, args.device)
    policy = load_policy(args.policy)
    out = args.out or os.path.join(ROOT, "results", os.path.splitext(os.path.basename(args.policy))[0] + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    results, start = [], time.time()
    for scene_id, scene in scenes.items():
        if args.scenes and scene_id not in args.scenes:
            continue
        for index, spec in enumerate(scene.episodes):
            video = None
            if args.save_video:
                os.makedirs(args.save_video, exist_ok=True)
                video = EpisodeVideo(os.path.join(args.save_video, f"{scene_id}_{index}.mp4"), scene, camera, policy, f"{scene_id} {index}")
            result, _ = run_episode(renderer, scene, spec, policy, on_step=video.add if video else None)
            if video:
                video.finish(result)
            result["episode"] = index
            results.append(result)
            print(f"{scene_id:28s} {index}  {'ok  ' if result['success'] else 'FAIL'}  completion {result['route_completion']:.2f}  "
                  f"collisions {result['collisions']}  {result['time']:5.1f}s  {result['reason']}   [{time.time() - start:.0f}s]", flush=True)

    summary = summarize(results)
    print("\n" + json.dumps(summary))
    with open(out, "w") as f:
        json.dump(dict(summary=summary, episodes=results), f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
