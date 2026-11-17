"""Fine-tune ViNT on the processed clips.

    python code/train.py --data data/processed --out runs/first

The data loading, the validation metric and the checkpointing are done. The training itself is yours: search for TODO.
Run the script before you change anything. It prints the validation error of the original model and of a trivial
baseline, which are the numbers to beat, and then stops at the first TODO.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F          # for the loss

from dataset import ClipDataset
from vint import STEP, forward, load_model, pick_device


@torch.no_grad()
def validate(model, loader, mean_path, dev):
    """Average distance (meters) between predicted and true waypoints, and the same for always predicting the average path."""
    model.eval()
    errors, baseline = [], []
    for obs, goal, target in loader:
        obs, goal, target = obs.to(dev), goal.to(dev), target.to(dev)
        pred = forward(model, obs, goal)
        errors.append(((pred - target).norm(dim=-1) * STEP).cpu())
        baseline.append(((mean_path.to(dev)[None] - target).norm(dim=-1) * STEP).cpu())
    return float(torch.cat(errors).mean()), float(torch.cat(baseline).mean())


def trainable_parameters(model, args):
    """Return the list of parameters that will be updated.

    TODO: decide what to train. Updating every weight is the obvious choice, but it is not the only one. The model was
    trained on other footage, so ask what should change to suit this robot and what should be kept. The parameters are named by part
    (print `[n for n, _ in model.named_parameters()]`): obs_encoder and goal_encoder are the image encoders, decoder is the transformer that reads their output.
    Try at least two settings and compare them in closed loop, not only by the validation error.
    """
    raise NotImplementedError("trainable_parameters")


def make_optimizer(params, args, steps):
    """Return (optimizer, scheduler). `steps` is the total number of optimizer steps, which a schedule may need.
    Call scheduler.step() once per optimizer step.

    TODO: choose the optimizer, the learning rate and the schedule. The weights are already good, and the learning rate
    that suits a pretrained network is not the one that suits a network trained from scratch. Too high a value destroys
    what the model knows; too low a value changes nothing.
    """
    raise NotImplementedError("make_optimizer")


def compute_loss(pred, target):
    """pred and target are [B, 5, 2]: the (x, y) of five waypoints in model units (one unit is 0.8 m).

    TODO: write the loss. Think about what an error in each waypoint costs the robot when it drives, and whether the
    far ones and the near ones should count equally. Return a scalar tensor.
    """
    raise NotImplementedError("compute_loss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto (CUDA if present, else CPU), cpu, cuda or mps")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = pick_device(args.device)

    model = load_model(device=dev)

    train_set = ClipDataset(args.data, "train")
    val_set = ClipDataset(args.data, "val", augment=False)
    # no shuffle=True: the training dataset already draws random samples
    train_loader = torch.utils.data.DataLoader(train_set, args.batch, num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(val_set, 128, num_workers=args.workers)
    print(f"{len(train_set.items)} training moments, {len(val_set.items)} validation moments, device {dev}")

    sample = np.stack([train_set[i][2].numpy() for i in range(2000)])
    mean_path = torch.from_numpy(sample.mean(0))

    ade, avg_ade = validate(model, val_loader, mean_path, dev)
    print(f"before training: validation error {ade:.3f} m (average path: {avg_ade:.3f} m)", flush=True)

    params = trainable_parameters(model, args)
    opt, sched = make_optimizer(params, args, args.epochs * len(train_loader))

    log, best = [dict(epoch=0, val_error=ade)], ade
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        for step, (obs, goal, target) in enumerate(train_loader):
            obs, goal, target = obs.to(dev), goal.to(dev), target.to(dev)

            # TODO: one optimization step: forward pass (see vint.forward), loss, backward, update.
            # Consider what keeps a pretrained network stable when a batch has a large error.
            raise NotImplementedError("training step")

            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
        ade, _ = validate(model, val_loader, mean_path, dev)
        print(f"epoch {epoch}: validation error {ade:.3f} m  ({time.time() - start:.0f} s)")
        log.append(dict(epoch=epoch, val_error=ade))
        torch.save(model.state_dict(), f"{args.out}/last.pt")
        if ade < best:
            best = ade
            torch.save(model.state_dict(), f"{args.out}/best.pt")
        json.dump(log, open(f"{args.out}/log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
