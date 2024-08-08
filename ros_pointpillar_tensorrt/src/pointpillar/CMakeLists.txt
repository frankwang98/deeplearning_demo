cmake_minimum_required(VERSION 2.8.7)
project(pointpillar)
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

if(${arch}     STREQUAL  "aarch64")
  # set(TENSORRT_ROOT         /root/.kiwi/lib/TensorRT-8.5.3.1-cuda11x)
  # set(CUDA_TOOLKIT_ROOT_DIR /root/.kiwi/lib/cuda-11.8)
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
endif()
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")
add_compile_options(-W)
add_compile_options(-std=c++11)

set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 87)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xcompiler -I/usr/aarch64-linux-gnu/include/
    -Xlinker -lsocket
    -Xlinker -rpath=/usr/lib/aarch64-linux-gnu/
    -Xlinker -rpath=/usr/aarch64-linux-gnu/lib/
    -Xlinker -L/usr/lib/aarch64-linux-gnu/
    -Xlinker -L/usr/aarch64-linux-gnu/lib/
)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

# set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
# set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)

# tensorrt
include_directories(/usr/local/TensorRT-8.5.1.7/include/)
link_directories(/usr/local/TensorRT-8.5.1.7/lib/)


find_package(CUDA REQUIRED)
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()
find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  jsk_recognition_msgs
)


if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

# set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
# set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    include
)

link_directories(
	${TENSORRT_LIBRARY_DIRS}
	/usr/lib/aarch64-linux-gnu
  /usr/aarch64-linux-gnu/lib/
)

file(GLOB_RECURSE SOURCE_FILES
    src/*.cu
    src/*.cpp
)
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros
)
cuda_add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES})


target_link_LIBraries(${PROJECT_NAME}
    libnvinfer.so
    libnvonnxparser.so
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
