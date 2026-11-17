"""Export a model to the ONNX file you submit.

    import sys; sys.path.append("code")
    from vint import Waypoints, load_model
    from export_onnx import export_module
    export_module(Waypoints(load_model("runs/mine/last.pt")), "submission.onnx")

`export_module` writes any PyTorch module to a file in the submission format and checks that ONNX Runtime reproduces it. Above, the module is ViNT
with your fine-tuned weights (a state_dict saved by train.py), wrapped by `Waypoints` in vint.py. For a ViNT that you changed, or any other
model, pass your own module: it takes (images, goal_image, goal) as described below and returns the waypoints, and it must do its own resizing
and normalizing (`Waypoints` shows how).

What the submitted file must do. The grader runs it with ONNX Runtime on the CPU (code/onnx_policy.py). The simulator gives it
ordinary camera images, not preprocessed ones, so resizing, normalizing and stacking the images is the file's own job.
The simulator runs at 4 Hz, one call per step, with a batch of one:

    input   "images"      uint8   [1, T, 126, 224, 3]   the last T camera images, oldest first: RGB, 224 wide and 126 high, with a
                                                        90 degree horizontal field of view. T is fixed by your file, from 1 to 25;
                                                        images come 0.25 s apart. At the start of an episode, when fewer than T
                                                        exist, the first image is repeated.
    input   "goal_image"  uint8   [1, 126, 224, 3]      the picture the camera would take at the end of the route, looking along it
    input   "goal"        float32 [1, 2]                the end of the route in meters: (forward, left) in the robot's frame
    output  "waypoints"   float32 [1, 5, 2]             where the robot should be 0.25, 0.5, 0.75, 1.0 and 1.25 seconds from now,
                                                        (forward, left) in meters in its current frame. The controller (policy.py)
                                                        takes the speed from the waypoint nearest the middle of that horizon and the
                                                        turn from the last one.

You may ignore an input (ViNT ignores "goal"), but the file must have all three. Check your file with
`python check_submission.py submission.onnx` before you submit it.
"""
import warnings

import numpy as np
import onnx
import torch
from onnx import helper

from vint import WAYPOINTS

IMAGE_SHAPE = (126, 224, 3)         # height, width, channels of the simulator's camera


def export_module(module, path, frames=6, opset=17):
    """Write any PyTorch module to `path` in the submission format, and check that ONNX Runtime reproduces it.

    The module is called as module(images, goal_image, goal) with images uint8 [1, frames, 126, 224, 3], goal_image uint8 [1, 126, 224, 3] and
    goal float32 [1, 2], and must return float32 waypoints [1, 5, 2] in meters. Resizing, normalizing and any other preparation of the pictures
    belong inside the module. `frames` is T, the number of camera images the model reads (1 to 25).
    """
    import onnxruntime

    module = module.cpu().eval()
    for part in module.modules():
        if hasattr(part, "set_swish"):                                  # EfficientNet's memory-saving activation cannot be exported
            part.set_swish(memory_efficient=False)
    rng = np.random.default_rng(0)
    images = torch.from_numpy(rng.integers(0, 256, (1, frames, *IMAGE_SHAPE), dtype=np.uint8))
    goal_image = torch.from_numpy(rng.integers(0, 256, (1, *IMAGE_SHAPE), dtype=np.uint8))
    goal = torch.tensor([[6.0, 0.0]])
    with torch.no_grad():
        expected = module(images, goal_image, goal)
    if tuple(expected.shape) != (1, WAYPOINTS, 2):
        raise ValueError(f"the module returns shape {tuple(expected.shape)}, expected (1, {WAYPOINTS}, 2)")

    kwargs = dict(input_names=["images", "goal_image", "goal"], output_names=["waypoints"], opset_version=opset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(module, (images, goal_image, goal), path, dynamo=False, **kwargs)       # the classic exporter
        except TypeError:                                                                           # older PyTorch without the `dynamo` argument
            torch.onnx.export(module, (images, goal_image, goal), path, **kwargs)

    model = onnx.load(path)
    present = {i.name for i in model.graph.input}
    for name, kind, shape in (("images", onnx.TensorProto.UINT8, [1, frames, *IMAGE_SHAPE]), ("goal_image", onnx.TensorProto.UINT8, [1, *IMAGE_SHAPE]),
                              ("goal", onnx.TensorProto.FLOAT, [1, 2])):
        if name not in present:                                         # the exporter drops inputs the model does not use; the file must still have them
            model.graph.input.append(helper.make_tensor_value_info(name, kind, shape))
    onnx.save_model(model, path)

    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["waypoints"], {"images": images.numpy(), "goal_image": goal_image.numpy(), "goal": goal.numpy()})[0]
    error = float(np.abs(got - expected.numpy()).max())
    print(f"wrote {path}; ONNX Runtime and PyTorch differ by at most {error:.1e} m on a test input")
    if error > 1e-3:
        raise RuntimeError("the exported model does not reproduce the PyTorch model")
