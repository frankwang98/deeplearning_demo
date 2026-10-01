# 端到端驾驶实验

这个目录把端到端学习拆成可以逐步验证的项目。第一阶段只使用车辆状态，先建立“专家采集数据 → 策略训练 → 闭环驾驶 → 失败分析”的完整链路；后续再把输入替换为图片和多传感器信息。

## 01：状态到控制

输入为横向偏差、航向误差、车速、当前道路曲率和前方曲率，模型直接输出方向盘转角与加速度。监督信号由一个可解释的专家控制器生成，这种方法称为行为克隆。

```bash
python projects/end_to_end_driving/01_state_to_control/train.py
python projects/end_to_end_driving/01_state_to_control/evaluate.py
```

训练输出保存在 `runs/end-to-end-state/`：

- `policy.pt`：模型、输入标准化参数和车辆配置。
- `training.json`：训练集与验证集损失。
- `evaluation.json`：闭环完成率、越界率和误差统计。
- `closed-loop.svg`：学习策略和专家策略的横向轨迹对比。

开环验证损失只说明模型在已有样本上接近专家。闭环评测会把模型输出重新作用于车辆；一个很小的误差也可能随时间累积，因此端到端驾驶必须看闭环结果。

## 02：图像到控制

第二阶段用轻量前视道路渲染器生成灰度图，CNN 只通过图像和车速输出控制。仍使用相同车辆动力学和道路进行闭环评测，因此可以公平比较状态策略与视觉策略。

```bash
python projects/end_to_end_driving/02_image_to_control/train.py
python projects/end_to_end_driving/02_image_to_control/evaluate.py
```

## 03：图像到轨迹点

第三阶段让网络预测车体系下 4、8、12 米处的未来横向轨迹点，几何控制器再把轨迹转换为方向盘角度。`evaluation.json` 会保存部分轨迹点快照，便于定位模型还是控制器的问题。

```bash
python projects/end_to_end_driving/03_image_to_waypoints/train.py
python projects/end_to_end_driving/03_image_to_waypoints/evaluate.py
```

## 04：外部数据契约

[`04_external_data`](04_external_data/README.md) 定义 CARLA 与 ROS2 数据进入训练流程前的 manifest 格式、验证工具和防止相邻帧泄漏的切分规则。

## 后续阶段

| 阶段 | 输入 | 输出 | 主要问题 |
| --- | --- | --- | --- |
| 01 | 车辆状态与道路曲率 | 转向、加速度 | 行为克隆与闭环评测 |
| 02 ✅ | 合成道路图像 | 转向与加速度 | 视觉表征与闭环漂移 |
| 03 ✅ | 图像与车速 | 未来轨迹点 | 可解释的混合端到端 |
| 04 ✅ | CARLA / ROS2 manifest | 统一训练样本 | 数据同步、切分和校验 |

完整学习顺序与指标说明见[端到端学习路线](../../docs/end-to-end-driving.md)。
