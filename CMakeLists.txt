cmake_minimum_required(VERSION 3.24)

project(VISION_CORE LANGUAGES CXX)

#cmake_minimum_required(VERSION 3.24)
#project(Vision_Core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# 可选：使用 vcpkg / 系统提供的 onnxruntime。当 vcpkg baseline 缺失该端口时保持 OFF，直接走本地 fallback。
option(ENABLE_VCPKG_ONNXRUNTIME "Attempt find_package(onnxruntime) before local fallback" OFF)
if(ENABLE_VCPKG_ONNXRUNTIME)
  find_package(onnxruntime CONFIG QUIET)
endif()
find_package(nlohmann_json CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_library(vision
  src/vision/Config.cpp
  src/vision/VisionA.cpp
  src/vision/OrtYolo.cpp
  src/vision/Nms.cpp
  src/vision/SeatRoi.cpp
  src/vision/Mog2.cpp
  src/vision/Snapshotter.cpp
  src/vision/Publish.cpp
  src/vision/Logging.cpp
)
target_include_directories(vision PUBLIC include)
target_link_libraries(vision
  PUBLIC
    opencv_core opencv_imgproc opencv_imgcodecs opencv_video
    nlohmann_json::nlohmann_json
    yaml-cpp::yaml-cpp
    spdlog::spdlog
)

add_executable(a_demo apps/a_demo.cpp)
target_link_libraries(a_demo PRIVATE vision)

add_executable(test_perf apps/test_perf.cpp)
target_link_libraries(test_perf PRIVATE vision)

add_executable(annotate_tables apps/annotate_tables.cpp)
target_link_libraries(annotate_tables PRIVATE vision)


# ---- ONNX Runtime 本地回退方案（不依赖 vcpkg 时启用） ----
# 若上面的 find_package 没有找到 onnxruntime 目标，则尝试从 thirdparty/onnxruntime 以导入库方式链接
if(NOT TARGET onnxruntime)
  set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/onnxruntime" CACHE PATH "ONNX Runtime root directory")
  # 兼容两种官方发布的头文件布局：
  # 1) include/onnxruntime_cxx_api.h
  # 2) include/onnxruntime/core/session/onnxruntime_cxx_api.h
  set(ONNXRUNTIME_INC_flat "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  set(ONNXRUNTIME_INC_nested "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
  if(EXISTS "${ONNXRUNTIME_INC_flat}")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_INC "${ONNXRUNTIME_INC_flat}")
  elseif(EXISTS "${ONNXRUNTIME_INC_nested}")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_INC "${ONNXRUNTIME_INC_nested}")
  else()
    set(ONNXRUNTIME_INCLUDE_DIR "")
  endif()

  set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
  set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")

  if(ONNXRUNTIME_INCLUDE_DIR AND EXISTS "${ONNXRUNTIME_LIB}")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_IMPLIB "${ONNXRUNTIME_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )
    if(WIN32 AND EXISTS "${ONNXRUNTIME_DLL}")
      set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_DLL}")
    endif()
    message(STATUS "Using local ONNX Runtime from ${ONNXRUNTIME_ROOT}")

    # 运行期复制 DLL（Windows）
    if(WIN32 AND EXISTS "${ONNXRUNTIME_DLL}")
      foreach(_exe a_demo test_perf)
        if(TARGET ${_exe})
          add_custom_command(TARGET ${_exe} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DLL}" $<TARGET_FILE_DIR:${_exe}>
            COMMENT "Copy onnxruntime.dll to runtime output for ${_exe}")
        endif()
      endforeach()
    endif()
  else()
    message(STATUS "[NOTICE] ONNX Runtime not found under ${ONNXRUNTIME_ROOT}; skipping ORT integration (fake_infer path active). (currently the ORT file is not yet generated thus if hard encoding, fatal error may occur.)")
  endif()
endif()


# 提示：实际目标（库/可执行）请在 src/ 与 apps/ 完成后补充。
# 先保持可配置通过，避免引用不存在的源文件导致错误。