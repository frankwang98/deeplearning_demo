"""Load the lesson-02 checkpoint and classify one two-dimensional point."""

import argparse
from pathlib import Path

import torch

from model import TinyClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/first-classifier/classifier.pt"))
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}; run train.py first")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TinyClassifier(saved["hidden_size"])
    model.load_state_dict(saved["model_state"])
    model.eval()
    point = torch.tensor([[args.x, args.y]], dtype=torch.float32)
    with torch.no_grad():
        probabilities = model((point - saved["mean"]) / saved["scale"]).softmax(dim=1)[0]
    class_index = probabilities.argmax().item()
    print(f"prediction={saved['class_names'][class_index]}")
    for name, probability in zip(saved["class_names"], probabilities.tolist()):
        print(f"  {name}: {probability:.2%}")


if __name__ == "__main__":
    main()
