cmake_minimum_required(VERSION 3.15)
project(SmartSeatingSystem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 查找依赖包
#find_package(PkgConfig REQUIRED)

# 查找 SQLiteCpp
#find_path(SQLITECPP_INCPLUDE_DIRS SQLiteCpp/SQLiteCpp.h)
#find_library(SQLITECPP_LIBRARIES SQLiteCpp)

# 查找 SQLite3
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(SQLITE3 REQUIRED sqlite3)

# 包含目录
include_directories(include)
#include_directories(${SQLITE3_INCLUDE_DIRS})
include_directories(tests)
include_directories(src)

# 添加可执行文件
add_executable(SmartSeatingSystem
    #src/main.cpp
    src/database/SeatDatabase.cpp
    src/database/DatabaseInitializer.cpp
    #src/analysis/Analyzer.cpp
    src/utils/TimeUtils.cpp
    #tests/test_seat_database.cpp
    #tests/test.cpp
    tests/DatabaseTest.cpp
)

# 链接库
target_link_libraries(SmartSeatingSystem
    #${SQLITECPP_LIBRARIES}
    #${SQLITE3_LIBRARIES}
    ${CMAKE_SOURCE_DIR}/lib/libSQLiteCpp.a
    ${CMAKE_SOURCE_DIR}/lib/libsqlite3.a
)

# 编译器选项
target_compile_options(SmartSeatingSystem PRIVATE -Wall -Wextra -O2)

# 创建数据库目录
add_custom_command(TARGET SmartSeatingSystem POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/database
)
