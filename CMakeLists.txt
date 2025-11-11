cmake_minimum_required(VERSION 3.23)

# 工程名
project(Library_System LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt6 基础设置（会自动开启 AUTOMOC/AUTOUIC/AUTORCC）
find_package(QT NAMES Qt6 REQUIRED COMPONENTS Core Widgets Charts)
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Charts)
qt_standard_project_setup()

# 递归收集你公开的头文件，便于 IDE 显示（含 include/seatui 根下的 *.hpp）
file(GLOB_RECURSE SEATUI_PUBLIC_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/include/seatui/**/*.hpp
)

# 可执行程序（启动器：登录 → 角色选择 → 学生/管理员）
qt_add_executable(Library_System
    WIN32

    # 源文件（实现 .cpp —— 注意：目录名是 launcher_app 不是 lacuner_app）
    src/launcher_app/main.cpp
    src/launcher_app/login_window.cpp
    src/launcher_app/role_selector.cpp
    src/student_app/student_window.cpp
    src/admin_app/admin_window.cpp
)

# 把公开头文件挂到目标上（IDE 可见，且 Q_OBJECT 的头更稳妥触发 moc）
target_sources(Library_System
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES ${SEATUI_PUBLIC_HEADERS}
)

# 头文件搜索路径（启用 #include <seatui/...>）
target_include_directories(Library_System
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# 链接 Qt 模块（Charts 必须显式链接，否则 QtCharts 类会“未声明/未解析”）
target_link_libraries(Library_System
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Charts
)

# 配置阶段打印一下 Qt 路径信息，便于排查
message(STATUS "Qt6 DIR         = ${Qt6_DIR}")
message(STATUS "Qt6 VERSION     = ${Qt6_VERSION}")
message(STATUS "Qt6Charts DIR   = ${Qt6Charts_DIR}")

# 部署（Qt Creator / 安装脚本）
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
