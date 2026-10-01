"""Validate the portable CARLA/ROS2 driving manifest contract."""

import argparse
import json
import math
from pathlib import Path


NUMERIC_FIELDS = ("speed_mps", "steering_rad", "acceleration_mps2")


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(path, check_files=False):
    previous_timestamp = -1
    first_timestamp = None
    sequence = None
    count = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
            timestamp = record.get("timestamp_ns")
            if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp <= previous_timestamp:
                raise ValueError(f"line {line_number}: timestamp_ns must be a strictly increasing integer")
            previous_timestamp = timestamp
            if first_timestamp is None:
                first_timestamp = timestamp
            current_sequence = record.get("sequence")
            if not isinstance(current_sequence, str) or not current_sequence:
                raise ValueError(f"line {line_number}: sequence must be a non-empty string")
            if sequence is None:
                sequence = current_sequence
            elif sequence != current_sequence:
                raise ValueError(f"line {line_number}: a manifest must contain exactly one sequence")
            image = record.get("image")
            if not isinstance(image, str) or not image or Path(image).is_absolute() or ".." in Path(image).parts:
                raise ValueError(f"line {line_number}: image must be a safe relative path")
            if check_files and not (path.parent / image).is_file():
                raise ValueError(f"line {line_number}: image does not exist: {image}")
            for field in NUMERIC_FIELDS:
                if not finite_number(record.get(field)):
                    raise ValueError(f"line {line_number}: {field} must be finite")
            waypoints = record.get("waypoints_m")
            if not isinstance(waypoints, list) or not waypoints:
                raise ValueError(f"line {line_number}: waypoints_m must be a non-empty list")
            previous_x = 0.0
            for waypoint in waypoints:
                if (not isinstance(waypoint, list) or len(waypoint) != 2
                        or not all(finite_number(value) for value in waypoint)
                        or waypoint[0] <= previous_x):
                    raise ValueError(
                        f"line {line_number}: waypoints must be finite [x, y] pairs with increasing x"
                    )
                previous_x = waypoint[0]
            count += 1
    if not count:
        raise ValueError("manifest contains no records")
    return {"sequence": sequence, "frames": count, "first_timestamp_ns": first_timestamp,
            "last_timestamp_ns": previous_timestamp}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    try:
        result = validate(args.manifest, args.check_files)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
