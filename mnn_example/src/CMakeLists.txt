add_library(MNNVisionEngine STATIC
  vision_engine.cpp
  common/common.cpp
  classifier/classifier.cpp
  face/face_engine.cpp
  object/object_engine.cpp
  face/aligner/aligner.cpp
  face/landmarker/landmarker.cpp
  face/recognizer/recognizer.cpp
  face/database/face_database.cpp
  face/detecter/detecter.cpp
  face/landmarker/pfldlandmarker/pfldlandmarker.cpp
  face/landmarker/zqlandmarker/zqlandmarker.cpp
  face/recognizer/mobilefacenet/mobilefacenet.cpp
  face/database/stream/file_stream.cpp
  face/detecter/ultraface/ultraface.cpp
  face/detecter/centerface/centerface.cpp
  object/mobilenetssd/mobilenetssd.cpp
)
target_compile_features(MNNVisionEngine PUBLIC cxx_std_17)
set_target_properties(MNNVisionEngine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(MNNVisionEngine PUBLIC ${OpenCV_INCLUDE_DIRS} ${DEMO_COMMON_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/.
  ${CMAKE_CURRENT_SOURCE_DIR}/classifier
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_SOURCE_DIR}/face
  ${CMAKE_CURRENT_SOURCE_DIR}/face/aligner
  ${CMAKE_CURRENT_SOURCE_DIR}/face/database
  ${CMAKE_CURRENT_SOURCE_DIR}/face/database/stream
  ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter
  ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/centerface
  ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/ultraface
  ${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker
  ${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/pfldlandmarker
  ${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/zqlandmarker
  ${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer
  ${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer/mobilefacenet
  ${CMAKE_CURRENT_SOURCE_DIR}/object
  ${CMAKE_CURRENT_SOURCE_DIR}/object/mobilenetssd
)
target_link_libraries(MNNVisionEngine PUBLIC ${OpenCV_LIBS} MNN::MNN)
if(MIRROR_OPENMP AND TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(MNNVisionEngine PUBLIC OpenMP::OpenMP_CXX)
endif()
if(MIRROR_BUILD_CLASSIFIER)
  add_executable(mnn_classifier ${DEMO_COMMON_DIR}/vision_demo.cpp)
  target_compile_definitions(mnn_classifier PRIVATE DEMO_MNN DEMO_CLASSIFIER)
  target_link_libraries(mnn_classifier PRIVATE MNNVisionEngine)
endif()
if(MIRROR_BUILD_OBJECT)
  add_executable(mnn_object ${DEMO_COMMON_DIR}/vision_demo.cpp)
  target_compile_definitions(mnn_object PRIVATE DEMO_MNN DEMO_OBJECT)
  target_link_libraries(mnn_object PRIVATE MNNVisionEngine)
endif()
if(MIRROR_BUILD_FACE)
  add_executable(mnn_face ${DEMO_COMMON_DIR}/vision_demo.cpp)
  target_compile_definitions(mnn_face PRIVATE DEMO_MNN DEMO_FACE)
  target_link_libraries(mnn_face PRIVATE MNNVisionEngine)
endif()
if(MIRROR_BUILD_LEGACY)
  add_executable(mnn_face_legacy ${CMAKE_CURRENT_SOURCE_DIR}/face/test_face.cpp)
  target_link_libraries(mnn_face_legacy PRIVATE MNNVisionEngine)
endif()
if(BUILD_TESTING)
  add_executable(mnn_lifecycle_tests ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/test_mnn.cpp)
  target_link_libraries(mnn_lifecycle_tests PRIVATE MNNVisionEngine)
  add_test(NAME mnn_invalid_models COMMAND mnn_lifecycle_tests)
endif()
