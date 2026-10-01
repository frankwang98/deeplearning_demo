"""Small deterministic road and bicycle-model simulator for the first driving project."""

from dataclasses import dataclass
import math

import torch


DT = 0.1
WHEELBASE = 2.7
MAX_STEERING = 0.5
MAX_ACCELERATION = 2.0
TARGET_SPEED = 8.0
OFF_ROAD_DISTANCE = 2.5


@dataclass
class VehicleState:
    distance: float = 0.0
    lateral_error: float = 0.0
    heading_error: float = 0.0
    speed: float = TARGET_SPEED


def road_curvature(distance):
    """Smooth road containing both long and short bends."""
    return 0.035 * torch.sin(distance / 15.0) + 0.018 * torch.sin(distance / 5.0)


def observation(state: VehicleState):
    current = float(road_curvature(torch.tensor(state.distance)))
    ahead = float(road_curvature(torch.tensor(state.distance + max(state.speed, 1.0) * 1.5)))
    return torch.tensor(
        [state.lateral_error, state.heading_error, state.speed, current, ahead],
        dtype=torch.float32,
    )


def expert_action(observations):
    """Stable feedback controller used only to create demonstrations and a baseline."""
    lateral = observations[..., 0]
    heading = observations[..., 1]
    speed = observations[..., 2]
    curvature = 0.65 * observations[..., 3] + 0.35 * observations[..., 4]
    feed_forward = torch.atan(WHEELBASE * curvature)
    steering = feed_forward - 0.32 * lateral - 0.95 * heading
    acceleration = 0.7 * (TARGET_SPEED - speed) - 0.3 * steering.abs() * speed
    return torch.stack(
        (steering.clamp(-MAX_STEERING, MAX_STEERING) / MAX_STEERING,
         acceleration.clamp(-MAX_ACCELERATION, MAX_ACCELERATION) / MAX_ACCELERATION),
        dim=-1,
    )


def step(state: VehicleState, normalized_action):
    steering = max(-1.0, min(1.0, float(normalized_action[0]))) * MAX_STEERING
    acceleration = max(-1.0, min(1.0, float(normalized_action[1]))) * MAX_ACCELERATION
    curvature = float(road_curvature(torch.tensor(state.distance)))
    state.speed = max(1.0, min(12.0, state.speed + acceleration * DT))
    state.heading_error += (
        state.speed / WHEELBASE * math.tan(steering) - curvature * state.speed
    ) * DT
    state.heading_error = max(-1.2, min(1.2, state.heading_error))
    state.lateral_error += state.speed * math.sin(state.heading_error) * DT
    state.distance += state.speed * math.cos(state.heading_error) * DT
    return state
