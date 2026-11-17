"""The ViNT navigation model: loading it, running it, and wrapping it into the interface the simulator and your submission use.

ViNT (General Navigation Models, Shah et al. 2023) is given the camera images of the last 1.25 seconds and a picture of the goal, and predicts
where the robot should be during the next 1.25 seconds. The network itself (vint_net.py) works on small, normalized inputs:

    input   obs     [B, 18, 64, 85]   six RGB frames, 0.25 s apart (4 Hz), 64 pixels high and 85 wide, normalized, stacked on the channel axis
            goal    [B, 3, 64, 85]    one RGB picture of the goal, the same size and normalization
    output  [B, 5, 2]                 the robot's position (x forward, y left) 0.25, 0.5, 0.75, 1.0 and 1.25 s from now, in the robot's
                                      current frame, in model units. One model unit is STEP meters.

ViNT was trained on a different camera than ours (the RECON robot). `align` maps a picture of our camera (224 x 126, 90 degree field of view) onto
what ViNT's camera would have seen, so that objects have the size and position the network expects.

`Waypoints` wraps everything into one module: camera images in (as the simulator provides them), waypoints in meters out. This is
what gets exported to the ONNX file you submit (see export_onnx.py). `forward` is the part you will use for training, on frames that
`prepare_data.py` has already aligned and normalized.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from vint_net import ViNT

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "vint.pt")      # the pretrained ViNT weights (the authors' checkpoint, as a state_dict)

STEP = 0.8              # meters per model unit
CONTEXT = 6             # camera frames the network reads
WAYPOINTS = 5           # positions it predicts
SPACING = 0.25          # seconds between the frames, and between the predicted positions
WIDTH, HEIGHT = 85, 64  # size of the network's pictures

MEAN = np.array([0.485, 0.456, 0.406], np.float32)      # ImageNet statistics: the image encoder was trained on pictures normalized this way
STD = np.array([0.229, 0.224, 0.225], np.float32)

# ViNT's training camera (RECON): focal lengths and principal point of its 640 x 440 pictures, scaled to 85 x 64.
FX = 272.547 * WIDTH / 640
FY = 266.358 * HEIGHT / 440
CX = 320.0 * WIDTH / 640
CY = 220.0 * HEIGHT / 440


def affine_for(width, height, focal):
    """The map from a pixel (x, y) of the network's 85 x 64 picture to a pixel of a source picture with a pinhole camera of the given
    size and focal length (in pixels): x_src = a * x + c, y_src = e * y + f. Returns (a, c, e, f), pixel centers at +0.5."""
    a, e = focal / FX, focal / FY
    return a, width / 2 - CX * a, e, height / 2 - CY * e


# the simulator's camera: 224 x 126, 90 degree field of view, so a focal length of 112 pixels
SIM_AFFINE = affine_for(224, 126, 112.0)


def pick_device(name="auto"):
    """CUDA if there is one, otherwise the CPU. The Apple GPU has to be requested explicitly with name="mps"."""
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----- the pictures ---------------------------------------------------------------------------------------------------------------

def align(images, source=SIM_AFFINE, size=(224, 126)):
    """Resample camera pictures to ViNT's camera. images: float [B, 3, H, W] in 0..1, with H x W = `size[::-1]`. Returns [B, 3, 64, 85].

    Bilinear sampling; whatever lies outside the source picture is black. This is the same map as PIL's Image.transform(..., Image.AFFINE).
    """
    a, c, e, f = source
    width, height = size
    xs = (torch.arange(WIDTH, dtype=torch.float32) + 0.5) * a + c           # source pixel coordinates of the output pixel centers
    ys = (torch.arange(HEIGHT, dtype=torch.float32) + 0.5) * e + f
    grid = torch.stack(torch.meshgrid(xs / width * 2 - 1, ys / height * 2 - 1, indexing="xy"), -1)       # [64, 85, 2], in -1..1
    grid = grid[None].expand(len(images), -1, -1, -1).to(images.device)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def normalize(pictures):
    """float [..., 3, H, W] in 0..1 -> normalized with the ImageNet statistics."""
    mean = torch.from_numpy(MEAN).to(pictures.device).view(3, 1, 1)
    std = torch.from_numpy(STD).to(pictures.device).view(3, 1, 1)
    return (pictures - mean) / std


# ----- the network ----------------------------------------------------------------------------------------------------------------

def load_model(checkpoint=None, device="cpu"):
    """ViNT with the pretrained weights from vint.pt, or with your fine-tuned weights if `checkpoint` is given (a state_dict saved
    with torch.save(model.state_dict(), path)). Returns the module on `device`."""
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"{WEIGHTS} is missing. Save nav-assets.tar in the nav-project folder and run `python unpack_files.py` in the mini-project-nav folder.")
    model = ViNT()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device)


def forward(model, obs, goal):
    """Run the network. obs [B, 18, 64, 85], goal [B, 3, 64, 85]. Returns the waypoints [B, 5, 2] in model units (one unit is STEP meters)."""
    return model(obs, goal)[1][:, :, :2]                                    # the network returns (distance to the goal, [positions, headings])


def prepare(images, goal_image):
    """Simulator pictures -> network inputs. images uint8 [1, T, 126, 224, 3] (the last CONTEXT are used, the oldest repeated if T is smaller),
    goal_image uint8 [1, 126, 224, 3]. Returns obs [1, 18, 64, 85] and goal [1, 3, 64, 85], aligned and normalized."""
    t = images.shape[1]
    index = [max(t - CONTEXT + k, 0) for k in range(CONTEXT)]
    frames = images[:, index].reshape(-1, *images.shape[2:]).permute(0, 3, 1, 2).float() / 255          # [6, 3, 126, 224]
    obs = normalize(align(frames)).reshape(1, CONTEXT * 3, HEIGHT, WIDTH)
    goal = normalize(align(goal_image.permute(0, 3, 1, 2).float() / 255))
    return obs, goal


class Waypoints(torch.nn.Module):
    """Camera images in, waypoints in meters out: what the simulator gives the policy, and what the submitted file does.

    images      uint8 [1, T, 126, 224, 3]   the last T camera images, oldest first, 0.25 s apart. T may be anything from 1 to 25;
                                            the network reads the last CONTEXT = 6 of them (the oldest one is repeated if T is smaller).
    goal_image  uint8 [1, 126, 224, 3]      what the camera sees from the end of the route
    goal        float32 [1, 2]              the end of the route in meters, (forward, left). ViNT does not use it; a model of yours may.
    returns     float32 [1, 5, 2]           where the robot should be 0.25, 0.5, ... 1.25 s from now, (forward, left) in meters
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, goal_image, goal):
        obs, target = prepare(images, goal_image)
        return forward(self.model, obs, target) * STEP
