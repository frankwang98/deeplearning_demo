# C++ 模型推理

C++ 示例统一为三类程序：`*_classifier`、`*_object`、`*_face`。每个程序接受图片、模型目录、输出图片和可选显示参数，错误会返回非零状态。

```bash
./mnn_classifier --image photo.jpg --models ./models --output result.jpg
./ncnn_object --image street.jpg --models ./models --output result.jpg --show
```

推理链路包括读取图片、按训练约定缩放与归一化、运行模型、解析输出、绘制并保存结果。输出的计时是这一整次 pipeline 调用，不等同于 GPU kernel benchmark。

根目录 CMake 可以只启用 MNN、只启用 NCNN或同时启用。`MIRROR_BUILD_CLASSIFIER`、`MIRROR_BUILD_OBJECT`、`MIRROR_BUILD_FACE` 可关闭不需要的目标，`MIRROR_BUILD_LEGACY=ON` 可编译历史入口。NCNN 的 Vulkan 支持需同时使用启用 Vulkan 的 NCNN 并设置 `MIRROR_VULKAN=ON`。
