"""Differentiable-free synthetic front-camera renderer used by stages 02 and 03."""

import torch


IMAGE_HEIGHT = 24
IMAGE_WIDTH = 32


def render_road(observations):
    """Render normalized observations as small grayscale front-camera images."""
    single = observations.ndim == 1
    if single:
        observations = observations.unsqueeze(0)
    device = observations.device
    rows = torch.linspace(0.0, 1.0, IMAGE_HEIGHT, device=device).view(1, IMAGE_HEIGHT, 1)
    columns = torch.linspace(-1.0, 1.0, IMAGE_WIDTH, device=device).view(1, 1, IMAGE_WIDTH)
    lateral = observations[:, 0].view(-1, 1, 1)
    heading = observations[:, 1].view(-1, 1, 1)
    curvature = observations[:, 3].view(-1, 1, 1)
    curvature_ahead = observations[:, 4].view(-1, 1, 1)

    depth = 1.0 - rows
    center = -0.24 * lateral - 0.9 * heading * depth - 3.8 * (
        0.65 * curvature + 0.35 * curvature_ahead
    ) * depth.square()
    half_width = 0.12 + 0.58 * depth
    line_width = 0.018 + 0.035 * depth
    left = torch.exp(-((columns - (center - half_width)) / line_width).square())
    right = torch.exp(-((columns - (center + half_width)) / line_width).square())
    road = ((columns > center - half_width) & (columns < center + half_width)).float() * 0.16
    horizon = 0.04 + 0.06 * rows
    images = (road + left + right + horizon).clamp(0.0, 1.0).unsqueeze(1)
    return images[0] if single else images
