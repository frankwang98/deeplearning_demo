# CUDA-PointPillars(TensorRT) with ROS1 Visualization on Jetson Orin
This project including the PointPillars using TensorRT and integrates it with ROS1 for real-time point cloud processing and visualization on Jetson Orin. 
## Prerequisites
ROS1 install: https://blog.csdn.net/lxr0106/article/details/136328659 or https://wiki.ros.org/noetic/Installation

Environment refers to CUDA-PointPillars: https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars

install jsk: ```sudo apt-get install ros-noetic-jsk-rviz-plugins``` **(ros-xxxx(your ros))**
## Installation Guide
1.```git clone https://github.com/wayyeah/PointPillarTensorRT-ROS.git```

2.**if TensorRT==8.4.x, ```run mv src/pointpillar/model/pointpillar.onnx src/pointpillar.onnx```** and **please modify the TensorRT path in src/pointpillars/CMakeLists.txt Line 64 65**

3.`` `cd PointPillarTensorRT-ROS```

4.```catkin_make```

<img width="651" alt="7" src="https://github.com/wayyeah/PointPillarTensorRT-ROS/assets/53206282/2b4dcc72-767e-40fd-990d-c041390f1b8d">


5.download KITTI bag: https://pan.baidu.com/s/14lB2Djw6iiivfuhaINgkyA?pwd=asr8 password: asr8

6.```roscore```#new terminate

7.```rosbag play kitti_2011_09_26_drive_0009_synced.bag```#new terminate

8.```source devel/setup.zsh``` #(zsh) 
or ```source devel/setup.sh ```#(bash) 

9.```rosrun pointpillar pointpillar```
if show the message ```trt_infer: 6: The engine plan file is generated on an incompatible device, expecting compute 7.2 got compute 8.7, please rebuild.```, please **delete pointpillar.onnx.cache** and run again

<img width="651" alt="2" src="https://github.com/wayyeah/PointPillarTensorRT-ROS/assets/53206282/bd3db6df-42b7-4b09-bfed-3a2dcf963c65">


10.```rviz```#new terminate

11.add topic 

<img width="651" alt="3" src="https://github.com/wayyeah/PointPillarTensorRT-ROS/assets/53206282/751b1aab-708d-4b55-8ff3-6166bc7f22fc">
<img width="651" alt="4" src="https://github.com/wayyeah/PointPillarTensorRT-ROS/assets/53206282/c0a70f04-d365-46b8-924b-a4c52c413209">
<img width="651" alt="5" src="https://github.com/wayyeah/PointPillarTensorRT-ROS/assets/53206282/cc7b0895-648a-4ced-873f-9d6fca8c0aae">


## Result

<img width="651" alt="5" src="https://github.com/wayyeah/PointPillarTensorRT-ROS/assets/53206282/2fd61278-3cee-47d6-86c1-e0879ceaa957">

