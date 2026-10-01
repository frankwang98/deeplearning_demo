"""Close the loop with predicted waypoints and a deterministic geometric controller."""

import argparse
import json
import random
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT / "01_state_to_control"))
sys.path.append(str(PROJECT / "02_image_to_control"))
from simulator import (MAX_ACCELERATION, MAX_STEERING, OFF_ROAD_DISTANCE, TARGET_SPEED,
                       WHEELBASE, VehicleState, observation, step)  # noqa: E402
from vision import render_road  # noqa: E402

from policy import WaypointPolicy


def load_policy(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = WaypointPolicy()
    model.load_state_dict(saved["model_state"])
    model.eval()

    def act(state):
        values = observation(state)
        image = render_road(values).unsqueeze(0)
        speed = ((values[2] - saved["speed_mean"]) / saved["speed_scale"]).reshape(1, 1)
        with torch.no_grad():
            waypoint_y = model(image, speed)[0] * saved["waypoint_scales"]
        predicted_curvature = (2.0 * waypoint_y / saved["lookaheads"].square()).mean()
        steering = torch.atan(WHEELBASE * predicted_curvature).clamp(-MAX_STEERING, MAX_STEERING)
        acceleration = (0.7 * (TARGET_SPEED - state.speed) - 0.3 * steering.abs() * state.speed).clamp(
            -MAX_ACCELERATION, MAX_ACCELERATION
        )
        return torch.stack((steering / MAX_STEERING, acceleration / MAX_ACCELERATION)), waypoint_y

    return act


def run_episode(policy, initial_state, max_steps, route_length):
    state = VehicleState(**vars(initial_state))
    errors, waypoint_snapshots = [], []
    off_road = False
    for index in range(max_steps):
        errors.append(abs(state.lateral_error))
        action, waypoints = policy(state)
        if index % 50 == 0:
            waypoint_snapshots.append(waypoints.tolist())
        step(state, action)
        if abs(state.lateral_error) > OFF_ROAD_DISTANCE:
            off_road = True
            break
        if state.distance >= route_length:
            break
    return {"completed": state.distance >= route_length, "off_road": off_road,
            "distance": state.distance, "mean_abs_lateral_error": sum(errors) / len(errors),
            "max_abs_lateral_error": max(errors), "waypoint_snapshots": waypoint_snapshots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/end-to-end-waypoints/waypoint-policy.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/end-to-end-waypoints"))
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--route-length", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=9)
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
    report = {"episodes": len(episodes), "route_length_m": args.route_length,
              "completion_rate": sum(x["completed"] for x in episodes) / len(episodes),
              "off_road_rate": sum(x["off_road"] for x in episodes) / len(episodes),
              "mean_abs_lateral_error": sum(x["mean_abs_lateral_error"] for x in episodes) / len(episodes),
              "max_abs_lateral_error": max(x["max_abs_lateral_error"] for x in episodes),
              "sample_waypoints": episodes[0]["waypoint_snapshots"]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sample_waypoints"}, indent=2))


if __name__ == "__main__":
    main()
