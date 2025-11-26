include(D:/CSC3002/Library_System/build/Desktop_Qt_6_7_3_MSVC2022_64bit-Debug/.qt/QtDeploySupport.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/Library_System-plugins.cmake" OPTIONAL)
set(__QT_DEPLOY_ALL_MODULES_FOUND_VIA_FIND_PACKAGE "ZlibPrivate;EntryPointPrivate;Core;Gui;Widgets;OpenGL;OpenGLWidgets;Charts;Network;WebSockets")

qt6_deploy_runtime_dependencies(
    EXECUTABLE D:/CSC3002/Library_System/build/Desktop_Qt_6_7_3_MSVC2022_64bit-Debug/Library_System.exe
    GENERATE_QT_CONF
)
