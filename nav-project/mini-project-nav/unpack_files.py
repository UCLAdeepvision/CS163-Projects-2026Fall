"""Unpack the model, the scenes and the training data where the code expects them.

    python unpack_files.py

Run it from this folder (mini-project-nav). It needs two files that you save in the nav-project folder (the folder that contains
mini-project-nav and navigation_colab.ipynb; on Colab, the nav-project folder in your Google Drive):

    nav-assets.tar        the pretrained model (code/vint.pt) and the scenes (evaluation-scenes/<scene>/...)   unpacked into this folder
    processed-data.tar    the training data (frames.npy, poses.npy, index.json)                                 unpacked into data/processed

Files that are already in place are skipped, so it is safe to run again. The tar files are left where they are.
"""
import argparse
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENES = ("8am", "dark", "hill", "round")
ASSET_FILES = [os.path.join("code", "vint.pt")] + [os.path.join("evaluation-scenes", s, f) for s in SCENES for f in ("splat.ply", "mesh.ply", "grid.npz")]


def find(name):
    """The path of the file `name` if it is next to this script or in the nav-project folder, otherwise None."""
    for folder in (ROOT, os.path.dirname(ROOT)):
        if os.path.exists(os.path.join(folder, name)):
            return os.path.join(folder, name)
    return None


def unpack(tar_path, destination):
    """Extract a tar file into `destination`, refusing paths that would land outside it."""
    os.makedirs(destination, exist_ok=True)
    base = os.path.realpath(destination)
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            target = os.path.realpath(os.path.join(base, member.name))
            if not (target == base or target.startswith(base + os.sep)) or member.issym() or member.islnk():
                sys.exit(f"{tar_path} contains an unsafe path ({member.name}); not unpacking it.")
        tar.extractall(base)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "processed"), help="where the training data is unpacked (default: data/processed)")
    args = ap.parse_args()
    missing = []

    if all(os.path.exists(os.path.join(ROOT, p)) for p in ASSET_FILES):
        print("the model and the scenes are in place")
    else:
        tar = find("nav-assets.tar")
        if tar is None:
            missing.append("nav-assets.tar")
        else:
            print(f"unpacking {tar} ...", flush=True)
            unpack(tar, ROOT)
            print("unpacked the model and the scenes")

    if os.path.exists(os.path.join(args.data, "frames.npy")):
        print("the training data is in place")
    else:
        tar = find("processed-data.tar")
        if tar is None:
            missing.append("processed-data.tar")
        else:
            print(f"unpacking {tar} ...", flush=True)
            unpack(tar, args.data)
            print("unpacked the training data into", args.data)

    if missing:
        sys.exit(f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not here. Save {'it' if len(missing) == 1 else 'them'} in the nav-project folder, "
                 f"{os.path.dirname(ROOT)}, then run this again.")
    print("done")


if __name__ == "__main__":
    main()
