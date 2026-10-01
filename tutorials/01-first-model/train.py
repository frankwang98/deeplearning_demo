"""Train y = wx + b on synthetic data; no GPU or dataset download needed."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/first-model"))
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs must be positive and lr must be finite and positive")

    # 1. Make reproducible data: the answer has noise, like real measurements.
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    x = torch.rand(256, 1) * 4 - 2
    y = 2 * x + 1 + 0.1 * torch.randn_like(x)
    train_x, val_x = x[:192], x[192:]
    train_y, val_y = y[:192], y[192:]

    # 2. The model initially knows neither the slope nor the intercept.
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    history = []
    with torch.no_grad():
        initial_val_loss = loss_fn(model(val_x), val_y).item()

    # 3. Forward -> loss -> gradients -> update; validation does not update weights.
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(train_x), train_y)
        if not torch.isfinite(train_loss):
            raise RuntimeError("Loss diverged; try a smaller --lr")
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()
        if not math.isfinite(val_loss):
            raise RuntimeError("Validation loss diverged; try a smaller --lr")
        history.append({"epoch": epoch, "train_loss": train_loss.item(), "val_loss": val_loss})
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:3d} train_loss={train_loss.item():.6f} val_loss={val_loss:.6f}")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "model.pt"
    torch.save(model.state_dict(), checkpoint)

    # 4. Inference uses a fresh model restored from the saved checkpoint.
    restored = nn.Linear(1, 1)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    restored.eval()
    with torch.no_grad():
        prediction = restored(torch.tensor([[3.0]])).item()
    report = {
        "torch_version": str(torch.__version__), "device": "cpu", "seed": args.seed,
        "epochs": args.epochs, "learning_rate": args.lr,
        "train_samples": len(train_x), "validation_samples": len(val_x),
        "initial_val_loss": initial_val_loss, "final_val_loss": history[-1]["val_loss"],
        "weight": restored.weight.item(), "bias": restored.bias.item(),
        "prediction_at_x3": prediction, "reference_at_x3": 7.0, "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Learned y = {report['weight']:.4f}x + {report['bias']:.4f}")
    print(f"Restored model: x=3 -> {prediction:.4f}; reference=7.0")
    print(f"Saved checkpoint and metrics to {args.output}")


if __name__ == "__main__":
    main()
