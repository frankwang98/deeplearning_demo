"""Learn steering and acceleration by cloning an expert controller."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from policy import DrivingPolicy
from simulator import expert_action


def create_demonstrations(sample_count: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    lateral = torch.empty(sample_count).uniform_(-2.2, 2.2, generator=generator)
    heading = torch.empty(sample_count).uniform_(-0.45, 0.45, generator=generator)
    speed = torch.empty(sample_count).uniform_(3.0, 11.0, generator=generator)
    curvature = torch.empty(sample_count).uniform_(-0.055, 0.055, generator=generator)
    curvature_ahead = (curvature + 0.018 * torch.randn(sample_count, generator=generator)).clamp(
        -0.065, 0.065
    )
    observations = torch.stack((lateral, heading, speed, curvature, curvature_ahead), dim=1)
    actions = expert_action(observations)
    order = torch.randperm(sample_count, generator=generator)
    return observations[order], actions[order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/end-to-end-state"))
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 100 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs and lr must be positive; samples must be at least 100")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    observations, actions = create_demonstrations(args.samples, args.seed)
    split = int(args.samples * 0.8)
    train_x, validation_x = observations[:split], observations[split:]
    train_y, validation_y = actions[:split], actions[split:]
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0).clamp_min(1e-6)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale

    model = DrivingPolicy()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss diverged; try a smaller --lr")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(validation_x), validation_y).item()
        history.append({"epoch": epoch, "train_loss": loss.item(),
                        "validation_loss": validation_loss})
        if epoch == 1 or epoch % 30 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:3d} train_loss={loss.item():.6f} "
                f"validation_loss={validation_loss:.6f}"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "mean": mean, "scale": scale,
         "hidden_size": 32, "feature_names": (
             "lateral_error", "heading_error", "speed", "curvature", "curvature_ahead")},
        args.output / "policy.pt",
    )
    report = {
        "torch_version": str(torch.__version__), "device": "cpu", "seed": args.seed,
        "samples": args.samples, "training_samples": split,
        "validation_samples": args.samples - split, "epochs": args.epochs,
        "learning_rate": args.lr, "final_validation_loss": validation_loss,
        "history": history,
    }
    (args.output / "training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved policy and training report to {args.output}")


if __name__ == "__main__":
    main()
