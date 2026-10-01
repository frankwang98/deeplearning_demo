"""Train a camera model to predict inspectable future road-center waypoints."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT / "01_state_to_control"))
sys.path.append(str(PROJECT / "02_image_to_control"))
from simulator import MAX_STEERING, WHEELBASE, expert_action  # noqa: E402
from vision import render_road  # noqa: E402

from policy import LOOKAHEADS, WaypointPolicy


def demonstrations(sample_count, seed):
    generator = torch.Generator().manual_seed(seed)
    observations = torch.stack(
        (torch.empty(sample_count).uniform_(-2.2, 2.2, generator=generator),
         torch.empty(sample_count).uniform_(-0.45, 0.45, generator=generator),
         torch.empty(sample_count).uniform_(3.0, 11.0, generator=generator),
         torch.empty(sample_count).uniform_(-0.055, 0.055, generator=generator),
         torch.empty(sample_count).uniform_(-0.065, 0.065, generator=generator)), dim=1
    )
    steering = expert_action(observations)[:, 0] * MAX_STEERING
    curvature = torch.tan(steering) / WHEELBASE
    waypoint_y = 0.5 * curvature.unsqueeze(1) * LOOKAHEADS.square().unsqueeze(0)
    scales = 0.5 * (math.tan(MAX_STEERING) / WHEELBASE) * LOOKAHEADS.square()
    order = torch.randperm(sample_count, generator=generator)
    return observations[order], (waypoint_y / scales)[order], scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output", type=Path, default=Path("runs/end-to-end-waypoints"))
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 100 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs and lr must be positive; samples must be at least 100")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    observations, targets, waypoint_scales = demonstrations(args.samples, args.seed)
    images = render_road(observations)
    split = int(args.samples * 0.8)
    speed_mean = observations[:split, 2].mean()
    speed_scale = observations[:split, 2].std().clamp_min(1e-6)
    speeds = ((observations[:, 2] - speed_mean) / speed_scale).unsqueeze(1)
    model = WaypointPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(images[:split], speeds[:split]), targets[:split])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(images[split:], speeds[split:]), targets[split:]).item()
        history.append({"epoch": epoch, "train_loss": loss.item(), "validation_loss": validation_loss})
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:3d} train_loss={loss.item():.6f} validation_loss={validation_loss:.6f}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "speed_mean": speed_mean, "speed_scale": speed_scale,
         "lookaheads": LOOKAHEADS, "waypoint_scales": waypoint_scales},
        args.output / "waypoint-policy.pt",
    )
    (args.output / "training.json").write_text(
        json.dumps({"seed": args.seed, "samples": args.samples, "epochs": args.epochs,
                    "final_validation_loss": validation_loss, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved waypoint policy and report to {args.output}")


if __name__ == "__main__":
    main()
