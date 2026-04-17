cmake_minimum_required(VERSION 3.14)
project(DistributionTool CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

set(SOURCES
    src/RadialBasis.cpp
    src/SphericalHarmonics.cpp
    src/SOAPDescriptor.cpp
    src/DefectClassifier.cpp
    src/Statistics.cpp
    src/PCA.cpp
    src/LammpsDumpReader.cpp
    src/main.cpp
)

add_executable(distool ${SOURCES})
target_include_directories(distool PRIVATE include)
target_link_libraries(distool PRIVATE Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(distool PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(distool PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-Wall -Wextra -g>
)
