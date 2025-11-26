cmake_minimum_required(VERSION 3.24)

project(Library_System LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 仅这一条；不要再写 find_package(QT NAMES Qt6 ...)
find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets Charts WebSockets)

# 自动开启 AUTOMOC/AUTOUIC/AUTORCC 等
qt_standard_project_setup()

# 可执行程序 —— 只放真正的 .cpp（均位于 src/**）
qt_add_executable(Library_System
    WIN32

    # 入口与启动流程
    src/launcher_app/main.cpp
    src/launcher_app/login_window.cpp
    src/launcher_app/role_selector.cpp

    # 学生端
    src/student_app/student_window.cpp
    src/student_app/navigation_canvas.cpp   # ← 注意：在 src/student_app 下

    # 管理端
    src/admin_app/admin_window.cpp

    # 公共小部件
    src/widgets/card_dialog.cpp
)

# 公开头文件集合（IDE 可见；基准目录为 include/）
# 说明：你说“navigation_canvas.hpp 在 student 下面和 student_window.hpp 并列”，
# 所以这里放在 include/seatui/student/ 下。
target_sources(Library_System
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES
      include/seatui/launcher/login_window.hpp
      include/seatui/launcher/role_selector.hpp
      include/seatui/student/student_window.hpp
      include/seatui/student/navigation_canvas.hpp
      include/seatui/admin/admin_window.hpp
      include/seatui/widgets/card_dialog.hpp
)

# 头文件检索路径，启用 <seatui/...>
target_include_directories(Library_System
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# 链接 Qt 模块
target_link_libraries(Library_System
  PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Charts
    Qt6::WebSockets
)

# 可选：输出一些调试信息（非必须）
message(STATUS "Qt6 DIR         = ${Qt6_DIR}")
message(STATUS "Qt6 VERSION     = ${Qt6_VERSION}")
message(STATUS "Qt6Charts DIR   = ${Qt6Charts_DIR}")

# 安装与部署（保留你的配置）
include(GNUInstallDirs)
install(TARGETS Library_System
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
qt_generate_deploy_app_script(
    TARGET Library_System
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
