# 常见问题

- **CMake 找不到 MNN**：确认安装目录含 `include/MNN/Interpreter.hpp` 和 `lib/libMNN.*`，设置 `-DMNN_ROOT=/实际路径`。
- **CMake 找不到 ncnn**：设置 `-Dncnn_DIR=/安装目录/lib/cmake/ncnn`，并确认该目录有 `ncnnConfig.cmake`。
- **模型初始化失败**：检查 [模型清单](models.md)中的文件名、权限和模型格式，先从分类程序排查。
- **类别数与标签不一致**：标签行数必须等于分类输出元素数；程序会明确报错，避免越界或静默显示错误标签。
- **图片读不到**：使用存在的 JPEG/PNG 路径；服务器上不要传 `--show`。
- **结果看起来错误**：先核对输入尺寸、BGR/RGB、均值、缩放系数和模型版本。这些参数属于模型契约。
- **PyTorch 安装慢或失败**：第一课使用 CPU 索引；确认 Python 版本受目标 PyTorch 支持，重新创建干净虚拟环境。
- **损失变成无穷或 NaN**：减小 `--lr`。第一课会主动停止并给出提示。
