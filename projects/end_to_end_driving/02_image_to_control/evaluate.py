"""Evaluate the camera policy in the same closed-loop vehicle simulator as stage 01."""

import argparse
import json
import random
from pathlib import Path
import sys

import torch

STAGE_ONE = Path(__file__).resolve().parents[1] / "01_state_to_control"
sys.path.append(str(STAGE_ONE))
from simulator import OFF_ROAD_DISTANCE, VehicleState, observation, step  # noqa: E402

from policy import VisionPolicy
from vision import render_road


def load_policy(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = VisionPolicy()
    model.load_state_dict(saved["model_state"])
    model.eval()

    def act(state):
        values = observation(state)
        image = render_road(values).unsqueeze(0)
        speed = ((values[2] - saved["speed_mean"]) / saved["speed_scale"]).reshape(1, 1)
        with torch.no_grad():
            return model(image, speed)[0]

    return act


def run_episode(policy, initial_state, max_steps, route_length):
    state = VehicleState(**vars(initial_state))
    errors = []
    off_road = False
    for _ in range(max_steps):
        errors.append(abs(state.lateral_error))
        step(state, policy(state))
        if abs(state.lateral_error) > OFF_ROAD_DISTANCE:
            off_road = True
            break
        if state.distance >= route_length:
            break
    return {"completed": state.distance >= route_length, "off_road": off_road,
            "distance": state.distance, "mean_abs_lateral_error": sum(errors) / len(errors),
            "max_abs_lateral_error": max(errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/end-to-end-vision/vision-policy.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/end-to-end-vision"))
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--route-length", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}; run train.py first")
    torch.set_num_threads(1)
    generator = random.Random(args.seed)
    starts = [VehicleState(lateral_error=generator.uniform(-1.6, 1.6),
                           heading_error=generator.uniform(-0.2, 0.2),
                           speed=generator.uniform(5.0, 10.0)) for _ in range(args.episodes)]
    policy = load_policy(args.checkpoint)
    episodes = [run_episode(policy, state, args.steps, args.route_length) for state in starts]
    report = {
        "episodes": len(episodes),
        "route_length_m": args.route_length,
        "completion_rate": sum(item["completed"] for item in episodes) / len(episodes),
        "off_road_rate": sum(item["off_road"] for item in episodes) / len(episodes),
        "mean_abs_lateral_error": sum(item["mean_abs_lateral_error"] for item in episodes) / len(episodes),
        "max_abs_lateral_error": max(item["max_abs_lateral_error"] for item in episodes),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
