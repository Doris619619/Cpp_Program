cmake_minimum_required(VERSION 3.10)
project(BModule_Standalone)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 适配Windows/Mac/Linux
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") # MSVC异常处理
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2") # Mac/Linux编译选项
endif()

# 1. 自动查找 OpenCV 4.x
find_package(OpenCV 4 REQUIRED)

# 2. json库配置
set(JSON_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/nlohmann")
if(NOT EXISTS "${JSON_INCLUDE_DIR}/json.hpp")
    message(FATAL_ERROR "json.hpp not found! 请下载到: ${JSON_INCLUDE_DIR}")
endif()

# 3. 头文件搜索路径（简化：OpenCV头文件自动获取，无需手动写路径）
include_directories(
    ${CMAKE_SOURCE_DIR}/include  # 你的头文件目录（seat_state_judger.hpp、data_structures.hpp）
    ${JSON_INCLUDE_DIR}          # json.hpp头文件
    ${OpenCV_INCLUDE_DIRS}       # OpenCV头文件（自动获取，无需硬编码）
    ${OpenCV_INCLUDE_DIRS}/opencv2
)

# 4. 源文件（保持不变，沿用你的原有配置）
add_executable(b_module_standalone
    src/main.cpp                 # 独立入口
    src/seat_state_judger.cpp    # B模块核心实现
    src/database/SeatDatabase.cpp # 编译数据库实现
)


target_link_libraries(b_module_standalone
    ${OpenCV_LIBS}  # OpenCV核心库（自动获取，包含core、imgproc、video等）
    pthread         # 线程库（用于this_thread::sleep_for）
)

# 6. 编译优化
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(b_module_standalone PRIVATE 
        -Wall        # 显示所有警告
        -Wextra      # 显示额外警告
        -O2          # 优化编译
        -Wno-unused  # 忽略“未使用变量”警告（可选）
    )
endif()

# 7. 打印配置信息（增强：显示OpenCV实际找到的版本和路径，便于排查）
message(STATUS "✅ Build Configuration Success!")
message(STATUS "C++ Standard: C++17")
message(STATUS "Include Paths:")
message(STATUS "  - 项目头文件: ${CMAKE_SOURCE_DIR}/include")
message(STATUS "  - OpenCV 版本: ${OpenCV_VERSION}")
message(STATUS "  - OpenCV 头文件: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "  - JSON: ${JSON_INCLUDE_DIR}")
message(STATUS "Output Executable: b_module_standalone (在build/目录下)")

# add_subdirectory(test)   #告诉CMake去处理test目录下的CMakeLists.txt