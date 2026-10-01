add_library(NCNNVisionEngine STATIC
  common/common.cpp
  classifier/classifier_engine.cpp
  face/face_engine.cpp
  object/object_engine.cpp
  classifier/classifier/mobilenet/mobilenet.cpp
  face/aligner/aligner.cpp
  face/landmarker/landmarker.cpp
  face/tracker/tracker.cpp
  face/recognizer/recognizer.cpp
  face/database/face_database.cpp
  face/detecter/detecter.cpp
  face/landmarker/insightface/insightface.cpp
  face/landmarker/zqlandmarker/zqlandmarker.cpp
  face/recognizer/mobilefacenet/mobilefacenet.cpp
  face/database/stream/file_stream.cpp
  face/detecter/anticonv/anticonv.cpp
  face/detecter/mtcnn/mtcnn.cpp
  face/detecter/retinaface/retinaface.cpp
  face/detecter/centerface/centerface.cpp
  object/object/mobilenetssd/mobilenetssd.cpp
)
target_compile_features(NCNNVisionEngine PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(NCNNVisionEngine PRIVATE -fexceptions)
endif()
set_target_properties(NCNNVisionEngine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(NCNNVisionEngine PUBLIC ${OpenCV_INCLUDE_DIRS} ${DEMO_COMMON_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/classifier
  ${CMAKE_CURRENT_SOURCE_DIR}/classifier/classifier/mobilenet
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_SOURCE_DIR}/face
  ${CMAKE_CURRENT_SOURCE_DIR}/face/aligner
  ${CMAKE_CURRENT_SOURCE_DIR}/face/database
  ${CMAKE_CURRENT_SOURCE_DIR}/face/database/stream
  ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter
  ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/anticonv
  ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/centerface
  ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/mtcnn
  ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/retinaface
  ${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker
  ${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/insightface
  ${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/zqlandmarker
  ${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer
  ${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer/mobilefacenet
  ${CMAKE_CURRENT_SOURCE_DIR}/face/tracker
  ${CMAKE_CURRENT_SOURCE_DIR}/object
  ${CMAKE_CURRENT_SOURCE_DIR}/object/object/mobilenetssd
)
target_link_libraries(NCNNVisionEngine PUBLIC ${OpenCV_LIBS} ncnn)
if(MIRROR_OPENMP AND TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(NCNNVisionEngine PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(NCNNVisionEngine PUBLIC MIRROR_STATIC)
if(MIRROR_VULKAN)
  if(NOT NCNN_VULKAN)
    message(FATAL_ERROR "MIRROR_VULKAN requires a Vulkan-enabled NCNN installation")
  endif()
  target_compile_definitions(NCNNVisionEngine PUBLIC MIRROR_VULKAN=1)
endif()
if(MIRROR_BUILD_CLASSIFIER)
  add_executable(ncnn_classifier ${DEMO_COMMON_DIR}/vision_demo.cpp)
  target_compile_definitions(ncnn_classifier PRIVATE DEMO_NCNN DEMO_CLASSIFIER)
  target_link_libraries(ncnn_classifier PRIVATE NCNNVisionEngine)
endif()
if(MIRROR_BUILD_OBJECT)
  add_executable(ncnn_object ${DEMO_COMMON_DIR}/vision_demo.cpp)
  target_compile_definitions(ncnn_object PRIVATE DEMO_NCNN DEMO_OBJECT)
  target_link_libraries(ncnn_object PRIVATE NCNNVisionEngine)
endif()
if(MIRROR_BUILD_FACE)
  add_executable(ncnn_face ${DEMO_COMMON_DIR}/vision_demo.cpp)
  target_compile_definitions(ncnn_face PRIVATE DEMO_NCNN DEMO_FACE)
  target_link_libraries(ncnn_face PRIVATE NCNNVisionEngine)
endif()
if(MIRROR_BUILD_LEGACY)
  add_executable(ncnn_face_legacy ${CMAKE_CURRENT_SOURCE_DIR}/../examples/test_face.cpp)
  target_link_libraries(ncnn_face_legacy PRIVATE NCNNVisionEngine)
endif()
if(MIRROR_INSTALL_SDK)
  install(TARGETS NCNNVisionEngine ARCHIVE DESTINATION lib)
  install(FILES common/common.h DESTINATION include/vision/common)
  foreach(component classifier object face)
    install(FILES ${component}/${component}_engine.h DESTINATION include/vision/${component})
  endforeach()
endif()
