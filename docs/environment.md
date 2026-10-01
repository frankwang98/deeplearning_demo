# 环境搭建

初学者先用 CPU 完成第一课。Python 3.10–3.12 均可，虚拟环境能避免不同案例互相污染。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tutorials/requirements.txt \
  --index-url https://download.pytorch.org/whl/cpu
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。只有数据规模或网络确实需要时再配置 CUDA，并从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择与你的系统、驱动匹配的命令。不要把 CUDA Toolkit 版本、显卡驱动支持的 CUDA 版本和 PyTorch wheel 标记当成同一个概念。

C++ 部署需要 CMake 3.16+、支持 C++17 的编译器和 OpenCV。MNN / NCNN 请独立编译安装，再通过 `MNN_ROOT` 或 `ncnn_DIR` 指向安装结果。TensorRT 项目与具体 CUDA、cuDNN、TensorRT 版本绑定，应读取子目录说明并记录自己的组合。
