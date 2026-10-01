"""Run learned and expert policies in closed loop and report driving metrics."""

import argparse
import html
import json
import random
from pathlib import Path

import torch

from policy import DrivingPolicy
from simulator import OFF_ROAD_DISTANCE, VehicleState, expert_action, observation, step


def load_policy(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = DrivingPolicy(saved["hidden_size"])
    model.load_state_dict(saved["model_state"])
    model.eval()

    def act(value):
        with torch.no_grad():
            return model((observation(value) - saved["mean"]) / saved["scale"])

    return act


def expert_policy(value):
    return expert_action(observation(value))


def run_episode(policy, initial_state, steps, route_length):
    state = VehicleState(**vars(initial_state))
    trace = []
    off_road = False
    for _ in range(steps):
        trace.append((state.distance, state.lateral_error))
        step(state, policy(state))
        if abs(state.lateral_error) > OFF_ROAD_DISTANCE:
            off_road = True
            break
        if state.distance >= route_length:
            break
    return {
        "completed": state.distance >= route_length,
        "off_road": off_road,
        "steps": len(trace),
        "distance": state.distance,
        "mean_abs_lateral_error": sum(abs(point[1]) for point in trace) / len(trace),
        "max_abs_lateral_error": max(abs(point[1]) for point in trace),
        "trace": trace,
    }


def summarize(episodes):
    return {
        "episodes": len(episodes),
        "completion_rate": sum(item["completed"] for item in episodes) / len(episodes),
        "off_road_rate": sum(item["off_road"] for item in episodes) / len(episodes),
        "mean_abs_lateral_error": sum(item["mean_abs_lateral_error"] for item in episodes)
        / len(episodes),
        "max_abs_lateral_error": max(item["max_abs_lateral_error"] for item in episodes),
        "mean_distance": sum(item["distance"] for item in episodes) / len(episodes),
    }


def write_svg(path, learned, expert):
    width, height = 900, 440
    margin = 55
    maximum_distance = max(point[0] for episode in learned + expert for point in episode["trace"])

    def polyline(trace):
        points = []
        for distance, lateral in trace:
            x = margin + distance / maximum_distance * (width - 2 * margin)
            y = height / 2 - lateral / OFF_ROAD_DISTANCE * (height / 2 - margin)
            points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    elements = [
        f'<rect width="{width}" height="{height}" fill="#f7fafc"/>',
        f'<line x1="{margin}" y1="{height / 2}" x2="{width - margin}" y2="{height / 2}" '
        'stroke="#2d3748" stroke-dasharray="8 6"/>',
    ]
    for boundary in (-OFF_ROAD_DISTANCE, OFF_ROAD_DISTANCE):
        y = height / 2 - boundary / OFF_ROAD_DISTANCE * (height / 2 - margin)
        elements.append(
            f'<line x1="{margin}" y1="{y}" x2="{width - margin}" y2="{y}" '
            'stroke="#e53e3e" stroke-width="2"/>'
        )
    for episode in expert[:5]:
        elements.append(
            f'<polyline points="{polyline(episode["trace"])}" fill="none" '
            'stroke="#718096" opacity=".55" stroke-width="1.5"/>'
        )
    for episode in learned[:5]:
        elements.append(
            f'<polyline points="{polyline(episode["trace"])}" fill="none" '
            'stroke="#3182ce" opacity=".8" stroke-width="2"/>'
        )
    elements.extend(
        [
            '<line x1="60" y1="24" x2="90" y2="24" stroke="#3182ce" stroke-width="3"/>',
            '<text x="98" y="29" fill="#172033" font-size="14">learned policy</text>',
            '<line x1="220" y1="24" x2="250" y2="24" stroke="#718096" stroke-width="3"/>',
            '<text x="258" y="29" fill="#172033" font-size="14">expert policy</text>',
            f'<text x="{width / 2}" y="{height - 10}" text-anchor="middle" '
            'fill="#4a5568" font-size="13">distance along road (m)</text>',
        ]
    )
    title = html.escape("Closed-loop lateral error; red lines are road boundaries")
    elements.append(f'<title>{title}</title>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(elements) + "</svg>\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/end-to-end-state/policy.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/end-to-end-state"))
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--route-length", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}; run train.py first")
    if args.episodes < 1 or args.steps < 1 or args.route_length <= 0:
        parser.error("episodes, steps and route length must be positive")

    torch.set_num_threads(1)
    random_generator = random.Random(args.seed)
    starts = [
        VehicleState(
            lateral_error=random_generator.uniform(-1.8, 1.8),
            heading_error=random_generator.uniform(-0.25, 0.25),
            speed=random_generator.uniform(5.0, 10.0),
        )
        for _ in range(args.episodes)
    ]
    learned_policy = load_policy(args.checkpoint)
    learned = [
        run_episode(learned_policy, state, args.steps, args.route_length) for state in starts
    ]
    expert = [run_episode(expert_policy, state, args.steps, args.route_length) for state in starts]
    report = {
        "seed": args.seed, "max_steps_per_episode": args.steps,
        "route_length_m": args.route_length,
        "road_half_width_m": OFF_ROAD_DISTANCE,
        "learned": summarize(learned), "expert": summarize(expert),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_svg(args.output / "closed-loop.svg", learned, expert)
    print(json.dumps(report, indent=2))
    print(f"Saved closed-loop report and visualization to {args.output}")


if __name__ == "__main__":
    main()
