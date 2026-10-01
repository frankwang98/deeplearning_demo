# CARLA / ROS2 数据接入契约

这一阶段定义外部数据进入训练代码前的统一格式。转换器应由实际使用的 CARLA 版本、ROS2 消息类型和 bag 格式决定；仓库提供稳定的数据契约与验证器，不声称在 CI 中运行 CARLA 或读取未提供的真实 bag。

每个序列包含一个 JSON Lines manifest，每行是一帧：

```json
{"timestamp_ns": 1000000000, "image": "camera/000001.png", "speed_mps": 7.8, "steering_rad": 0.03, "acceleration_mps2": 0.1, "waypoints_m": [[4.0, 0.2], [8.0, 0.5], [12.0, 1.0]], "sequence": "town01_route03"}
```

运行验证：

```bash
python projects/end_to_end_driving/04_external_data/validate_manifest.py path/to/manifest.jsonl
python projects/end_to_end_driving/04_external_data/validate_manifest.py path/to/manifest.jsonl --check-files
```

要求：时间戳严格递增；速度和动作必须是有限数值；至少有一个未来轨迹点，轨迹纵向距离严格递增；同一文件只能属于一个序列。`--check-files` 还会检查图片是否存在，图片路径相对于 manifest 所在目录解析。

## 数据源映射

| 统一字段 | CARLA 示例 | ROS2 示例 |
| --- | --- | --- |
| `timestamp_ns` | sensor frame timestamp | message header stamp |
| `image` | RGB camera frame | `sensor_msgs/Image` 导出文件 |
| `speed_mps` | vehicle velocity norm | odometry / vehicle status |
| `steering_rad` | expert vehicle control | control command / report |
| `acceleration_mps2` | throttle/brake 转换值 | IMU 或纵向控制命令 |
| `waypoints_m` | route planner future points | planning trajectory 转到车体系 |
| `sequence` | town + route + weather run | bag / route / recording ID |

训练、验证、测试必须按 `sequence` 切分，不能随机打散相邻帧后再切分，否则几乎相同的连续画面会造成数据泄漏。还应保存相机内外参、车辆坐标约定、软件版本与采集配置作为序列级 metadata。
