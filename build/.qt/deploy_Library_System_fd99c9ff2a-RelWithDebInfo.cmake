include(D:/CSC3002/Library_System/build/.qt/QtDeploySupport-RelWithDebInfo.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/Library_System-plugins-RelWithDebInfo.cmake" OPTIONAL)
set(__QT_DEPLOY_ALL_MODULES_FOUND_VIA_FIND_PACKAGE "ZlibPrivate;EntryPointPrivate;Core;Gui;Widgets;OpenGL;OpenGLWidgets;Charts")

qt6_deploy_runtime_dependencies(
    EXECUTABLE D:/CSC3002/Library_System/build/RelWithDebInfo/Library_System.exe
    GENERATE_QT_CONF
)
