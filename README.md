# pytorch_perception
AI模型训练、测试、部署。

之前只是零零散散写过一点东西，这次整理出来，记录自己遇到的一些坑。

## 环境
1. Windows-Python   
训练和测试在Windows下，Miniconda+CUDA12.0+Pytorch，Pytorch安装指令：`pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121` （安装GPU版本有问题）   
大多数github仓库都有requirements依赖，只需安装即可：`pip install -r requirements.txt`   
pip安装失败时，注意要把梯子先关掉。

2. Ubuntu-Cpp   
这部分包括模型的转换（onnx ncnn pth weights），以及用TensorRT如何去推理等。