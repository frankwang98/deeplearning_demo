"""Train a visible three-class classifier without downloading a dataset."""

import argparse
import html
import json
import math
from pathlib import Path

import torch
from torch import nn

from model import CLASS_NAMES, TinyClassifier


def make_dataset(seed: int):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-1.5, -1.0], [1.5, -0.8], [0.0, 1.6]])
    features = []
    labels = []
    for class_index, center in enumerate(centers):
        features.append(center + 0.42 * torch.randn(120, 2, generator=generator))
        labels.append(torch.full((120,), class_index, dtype=torch.long))
    features = torch.cat(features)
    labels = torch.cat(labels)
    order = torch.randperm(len(features), generator=generator)
    return features[order], labels[order]


def confusion_matrix(expected, predicted):
    matrix = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)
    for truth, guess in zip(expected.tolist(), predicted.tolist()):
        matrix[truth, guess] += 1
    return matrix.tolist()


def write_svg(path, model, mean, scale, features, labels):
    width = height = 540
    padding = 40
    minimum, maximum = -2.8, 2.8
    colors = ("#ffb454", "#68d391", "#63b3ed")

    def screen(point):
        x = padding + (point[0] - minimum) / (maximum - minimum) * (width - 2 * padding)
        y = height - padding - (point[1] - minimum) / (maximum - minimum) * (height - 2 * padding)
        return x, y

    cells = []
    grid_size = 46
    coordinates = torch.linspace(minimum, maximum, grid_size)
    grid = torch.cartesian_prod(coordinates, coordinates)
    with torch.no_grad():
        classes = model((grid - mean) / scale).argmax(dim=1)
    cell = (width - 2 * padding) / (grid_size - 1) + 1
    for point, class_index in zip(grid.tolist(), classes.tolist()):
        x, y = screen(point)
        cells.append(
            f'<rect x="{x - cell / 2:.1f}" y="{y - cell / 2:.1f}" '
            f'width="{cell:.1f}" height="{cell:.1f}" fill="{colors[class_index]}" opacity=".22"/>'
        )
    points = []
    for point, class_index in zip(features.tolist(), labels.tolist()):
        x, y = screen(point)
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{colors[class_index]}" '
            'stroke="#172033" stroke-width=".8"/>'
        )
    legend = []
    for index, name in enumerate(CLASS_NAMES):
        x = 42 + index * 145
        legend.append(f'<circle cx="{x}" cy="22" r="6" fill="{colors[index]}"/>')
        legend.append(
            f'<text x="{x + 10}" y="27" fill="#172033" font-size="14">{html.escape(name)}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#f7fafc"/>'
        + "".join(cells)
        + f'<rect x="{padding}" y="{padding}" width="{width - 2 * padding}" '
        f'height="{height - 2 * padding}" fill="none" stroke="#718096"/>'
        + "".join(points)
        + "".join(legend)
        + "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/first-classifier"))
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs must be positive and lr must be finite and positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    features, labels = make_dataset(args.seed)
    train_features, validation_features = features[:270], features[270:]
    train_labels, validation_labels = labels[:270], labels[270:]
    mean = train_features.mean(dim=0)
    scale = train_features.std(dim=0).clamp_min(1e-6)
    train_features = (train_features - mean) / scale
    validation_features = (validation_features - mean) / scale

    model = TinyClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(train_features), train_labels)
        if not torch.isfinite(train_loss):
            raise RuntimeError("Loss diverged; try a smaller --lr")
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_features)
            validation_loss = loss_fn(validation_logits, validation_labels).item()
            predictions = validation_logits.argmax(dim=1)
            accuracy = (predictions == validation_labels).float().mean().item()
        history.append(
            {"epoch": epoch, "train_loss": train_loss.item(),
             "validation_loss": validation_loss, "validation_accuracy": accuracy}
        )
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:3d} train_loss={train_loss.item():.5f} "
                f"val_loss={validation_loss:.5f} val_accuracy={accuracy:.1%}"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "classifier.pt"
    torch.save(
        {"model_state": model.state_dict(), "mean": mean, "scale": scale,
         "class_names": CLASS_NAMES, "hidden_size": 16},
        checkpoint,
    )
    matrix = confusion_matrix(validation_labels, predictions)
    report = {
        "torch_version": str(torch.__version__), "device": "cpu", "seed": args.seed,
        "epochs": args.epochs, "learning_rate": args.lr,
        "train_samples": len(train_features), "validation_samples": len(validation_features),
        "final_validation_loss": validation_loss, "validation_accuracy": accuracy,
        "class_names": CLASS_NAMES, "confusion_matrix": matrix, "history": history,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_svg(args.output / "decision-boundary.svg", model, mean, scale, features, labels)
    print("Confusion matrix (rows=true, columns=predicted):")
    for name, row in zip(CLASS_NAMES, matrix):
        print(f"  {name}: {row}")
    print(f"Saved checkpoint, metrics and decision boundary to {args.output}")


if __name__ == "__main__":
    main()
