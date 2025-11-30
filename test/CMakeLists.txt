# 最低CMake版本要求（与主工程一致）
cmake_minimum_required(VERSION 3.10)

# 测试程序名称
project(seat_state_test)

# 设置C++标准（与主工程一致，如C++17）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 新增：查找OpenCV（与主工程保持一致）
find_package(OpenCV 4 REQUIRED)

# 包含头文件目录（核心头文件 + 第三方库）
include_directories(
    ${PROJECT_SOURCE_DIR}/../include  # 你的核心头文件（seat_state_judger.hpp等）
    ${PROJECT_SOURCE_DIR}/../third_party  # nlohmann/json所在目录
    ${OpenCV_INCLUDE_DIRS}  # 新增：OpenCV头文件目录
)

# 查找核心业务代码（src目录下的cpp文件）
file(GLOB_RECURSE CORE_SOURCES ${PROJECT_SOURCE_DIR}/../src/*.cpp)
# 排除main.cpp（避免重复main函数冲突）
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

# 测试程序源代码
add_executable(seat_state_test
    test_seat_states.cpp  # 测试主程序
    ${CORE_SOURCES}       # 核心业务代码（SeatStateJudger等）
)

# 新增：链接OpenCV库和线程库（与主工程保持一致）
target_link_libraries(seat_state_test
    ${OpenCV_LIBS}
    pthread
)

# 输出测试程序到build目录下的test文件夹
set_target_properties(seat_state_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../build/test
)