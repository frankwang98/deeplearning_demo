# 模型文件说明

模型权重通常较大且带有各自许可证，本仓库不重复提交。请核对来源、哈希、输入尺寸、颜色顺序和归一化参数。

| 程序 | 模型目录内的主要文件 |
| --- | --- |
| MNN 分类 | `mobilenet.mnn`、`label.txt` |
| MNN 目标检测 | `mobilenetssd.mnn` |
| MNN 人脸 | `RFB-320.mnn`、`zqlandmark.mnn`、`mobilefacenet.mnn` |
| NCNN 分类 | `mobilenet.param`、`mobilenet.bin`、`label.txt` |
| NCNN 目标检测 | `mobilenetssd.param`、`mobilenetssd.bin` |
| NCNN 人脸 | `fd.param/bin`、`2d106.param/bin`、`fr.param/bin` |

MNN 历史说明包含 TensorFlow MobileNet 与 Caffe MobileNetSSD 的转换命令；NCNN 历史说明保留原模型下载链接，见对应目录 `README.legacy.md`。`label.txt` 必须与模型输出类别数一致，支持每行纯标签、数字索引前缀或 ImageNet synset 前缀。
