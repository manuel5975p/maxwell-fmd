
cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)
project(fmd)
find_package(SFML 2.5.1 COMPONENTS graphics system window QUIET)
if(NOT SFML_FOUND)
    option(BUILD_SHARED_LIBS     "" FALSE)
    option(SFML_STATIC_LIBRARIES "" TRUE )
    option(SFML_BUILD_EXAMPLES   "" FALSE)
    option(SFML_BUILD_DOC        "" FALSE)
    option(SFML_BUILD_WINDOW     "" TRUE )
    option(SFML_BUILD_GRAPHICS   "" TRUE )
    option(SFML_BUILD_AUDIO      "" FALSE)
    option(SFML_BUILD_NETWORK    "" FALSE)
    FetchContent_Declare(
        SFML
        GIT_REPOSITORY https://github.com/SFML/SFML.git
        GIT_TAG        2.5.1
    )
    FetchContent_MakeAvailable(SFML)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -march=native")
find_package (Eigen3 3.3 QUIET)
if (NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    )
    FetchContent_MakeAvailable(Eigen)
endif()
add_executable            (straightforward "fmdmaxwell.cpp")
target_link_libraries     (straightforward sfml-system)
target_link_libraries     (straightforward sfml-window)
target_link_libraries     (straightforward sfml-graphics)
target_link_libraries     (straightforward gomp)
target_link_libraries     (straightforward Eigen3::Eigen)
