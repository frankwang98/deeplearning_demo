# deeplearning_demo

一个从“训练第一个模型”走到“在 C++ / ROS 中部署模型”的 AI 实践仓库。这里保留了作者过去在本地真实测试过的 PyTorch、MNN、NCNN 和 TensorRT 案例，并补上适合初学者的路线、统一命令行、现代 CMake 和自动检查。

> 理论知识地图见 [awesome_hub](https://github.com/frankwang98/awesome_hub)。本仓库对应其中“感知 → 部署”的动手实验；两者可以配合学习。

## 从哪里开始

| 阶段 | 你会学到 | 实践入口 |
| --- | --- | --- |
| 0. 准备环境 | Python、虚拟环境、CPU/GPU 的区别 | [环境搭建](docs/environment.md) |
| 1. 第一个模型 | 张量、训练集、损失、梯度、验证、推理 | [10 分钟训练线性模型](docs/first-model.md) |
| 2. 图像分类 | 数据加载、分类与神经网络 | `pytorch_example/01-basics` |
| 3. 视觉网络 | CNN、RNN、ResNet、GAN、VAE | `pytorch_example/02-intermediate`、`03-advanced` |
| 4. 自有数据 | 训练和加载嘴部分类模型 | `pytorch_example/lenet_mouth` |
| 5. C++ 推理 | 模型文件、前后处理、MNN / NCNN | [C++ 部署](docs/cpp-inference.md) |
| 6. 工程部署 | TensorRT、YOLO、BEV、ROS | `tensorrt_example`、`pytorch_example/yolov5`、`yolop` |

完整顺序、每阶段目标和练习见 [AI 学习路线](docs/learning-path.md)。第一次学习建议只运行第一课，不要一次安装所有子项目的依赖。

## 10 分钟跑通第一课

```bash
python3 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -r tutorials/01-first-model/requirements.txt \
  --index-url https://download.pytorch.org/whl/cpu
python tutorials/01-first-model/train.py
```

程序会学习接近 `y = 2x + 1` 的规律，在 `runs/first-model/` 保存模型与训练指标，再重新加载模型预测 `x=3`。逐行讲解和实验题见 [第一课](docs/first-model.md)。

## C++ 推理快速入口

公共构建入口默认只编译不依赖推理框架的回归测试：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

选择一个后端，再给出已经安装的 SDK：

```bash
cmake -S . -B build-mnn -DDEMO_BUILD_MNN=ON -DMNN_ROOT=/opt/MNN
cmake --build build-mnn -j
./build-mnn/mnn_example/src/mnn_classifier \
  --image photo.jpg --models /path/to/mnn-models --output result.jpg

cmake -S . -B build-ncnn -DDEMO_BUILD_NCNN=ON -Dncnn_DIR=/opt/ncnn/lib/cmake/ncnn
cmake --build build-ncnn -j
./build-ncnn/ncnn_example/src/ncnn_classifier \
  --image photo.jpg --models /path/to/ncnn-models --output result.jpg
```

命令默认保存结果，只有传 `--show` 才打开窗口，适合服务器和 CI。模型清单与转换来源见 [模型文件说明](docs/models.md)，故障排查见 [常见问题](docs/troubleshooting.md)。

## 仓库结构

- `tutorials/`：为初学者新增的、可独立运行的小实验。
- `pytorch_example/`：作者历史训练示例，以及 YOLOP / YOLOv5 等完整项目。
- `mnn_example/`、`ncnn_example/`：轻量 C++ 端侧推理。
- `tensorrt_example/`：TensorRT、BEV、PointPillars 和 ROS 部署案例。
- `common/`、`tests/`：统一 CLI、标签解析及回归测试。

## 验证范围

原有示例来自作者过去的本地真实测试，旧说明保存在各模块的 `README.legacy.md`。本次维护实际验证了第一课 CPU 训练、公共回归测试，以及 MNN 2.9.3 的完整编译与生命周期测试；深层目录包含若干上游项目，各自依赖、模型和硬件不同，需按对应目录说明运行。CI 负责可重复检查代码格式、公共 C++ 测试与第一课训练，不会假装验证没有模型权重或 GPU 的推理结果。

## 参与维护

新增教程时请说明学习目标、前置知识、运行命令、预期现象和练习；新增部署案例时请记录框架版本、模型来源、预处理、输出格式与验证环境。C++ 提交运行 `clang-format -i`，仓库格式由 `.clang-format` 统一。
