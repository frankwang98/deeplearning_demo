"""Train a small CNN to map a synthetic road image directly to vehicle controls."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn

STAGE_ONE = Path(__file__).resolve().parents[1] / "01_state_to_control"
sys.path.append(str(STAGE_ONE))
from simulator import expert_action  # noqa: E402

from policy import VisionPolicy
from vision import render_road


def demonstrations(sample_count, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.stack(
        (
            torch.empty(sample_count).uniform_(-2.2, 2.2, generator=generator),
            torch.empty(sample_count).uniform_(-0.45, 0.45, generator=generator),
            torch.empty(sample_count).uniform_(3.0, 11.0, generator=generator),
            torch.empty(sample_count).uniform_(-0.055, 0.055, generator=generator),
            torch.empty(sample_count).uniform_(-0.065, 0.065, generator=generator),
        ),
        dim=1,
    )
    order = torch.randperm(sample_count, generator=generator)
    return values[order], expert_action(values[order])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, default=Path("runs/end-to-end-vision"))
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 100 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs and lr must be positive; samples must be at least 100")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    observations, actions = demonstrations(args.samples, args.seed)
    images = render_road(observations)
    speed_mean = observations[: int(args.samples * 0.8), 2].mean()
    speed_scale = observations[: int(args.samples * 0.8), 2].std().clamp_min(1e-6)
    speeds = ((observations[:, 2] - speed_mean) / speed_scale).unsqueeze(1)
    split = int(args.samples * 0.8)
    model = VisionPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(images[:split], speeds[:split]), actions[:split])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(images[split:], speeds[split:]), actions[split:]).item()
        history.append({"epoch": epoch, "train_loss": loss.item(), "validation_loss": validation_loss})
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:3d} train_loss={loss.item():.6f} validation_loss={validation_loss:.6f}")

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "speed_mean": speed_mean,
         "speed_scale": speed_scale, "image_shape": tuple(images.shape[1:])},
        args.output / "vision-policy.pt",
    )
    (args.output / "training.json").write_text(
        json.dumps({"seed": args.seed, "samples": args.samples, "epochs": args.epochs,
                    "final_validation_loss": validation_loss, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved camera policy and report to {args.output}")


if __name__ == "__main__":
    main()
