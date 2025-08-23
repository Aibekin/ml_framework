cmake_minimum_required(VERSION 3.10)
project(MachineLearning LANGUAGES C CXX)

set(CMAKE_PREFIX_PATH "C:/Users/User/Desktop/programming/libraries/matrix")

find_package(Matrix REQUIRED)

file(GLOB_RECURSE MachineLearning_SOURCES CONFIGURE_DEPENDS src/*.c include/*.h)
add_library(MachineLearning SHARED ${MachineLearning_SOURCES})

target_include_directories(MachineLearning PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(MachineLearning PUBLIC Matrix::Matrix)

add_library(MachineLearning::MachineLearning ALIAS MachineLearning)

include(GNUInstallDirs)

install(TARGETS MachineLearning
    EXPORT MachineLearningTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT MachineLearningTargets
    FILE MachineLearningTargets.cmake
    NAMESPACE MachineLearning::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MachineLearning
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MachineLearningConfigVersion.cmake
    VERSION 1.0
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MachineLearningConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MachineLearningConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MachineLearning
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MachineLearningConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MachineLearningConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MachineLearning
)