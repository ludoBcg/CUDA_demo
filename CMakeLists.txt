######################################################################################################
#
# CMakeLists.txt
#
# CUDA_demo
# Ludovic Blache
#
######################################################################################################



# check CMAKE version
cmake_minimum_required(VERSION 3.8)

# PATH TO EXTERNAL LIBS
set(EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")

# create project
project(CUDA_demo LANGUAGES CXX CUDA)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)

# Enable OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add source files
set(SRCS
    src/basics.cu
    src/convolution.cu
    src/texture.cu
    src/comparison.cpp
    src/main.cpp
    )
    

set(HEADERS
    src/utils.h
    src/comparison.h
    src/basics.h
    src/convolution.h
    src/texture.h
    )


######################################################################################################
# INTERNAL DEPENDENCIES
######################################################################################################


# Cuda ---------------------------------------------------------------
SET(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib/x64)
SET(CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} cudart.lib)

# lodepng
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng" PROJECT_SRCS)
include_directories(SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng")


######################################################################################################
# OPTIONAL (to test OpenGL interoperability only)
######################################################################################################

# OpenGL
find_package(OpenGL REQUIRED)
if(OPENGL_FOUND)
  include_directories(SYSTEM ${OPENGL_INCLUDE_DIR})
  #set(PROJECT_LIBRARIES ${PROJECT_LIBRARIES} ${OPENGL_LIBRARIES})
endif(OPENGL_FOUND)

# GLEW (download binaries for windows)
set(GLEW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/glew-2.1.0")
include_directories(${GLEW_DIR}/include)
link_directories(${GLEW_DIR}/lib/Release/x64)
# use glew32s only
SET(GLEW_LIBS glew32s.lib)
add_definitions(-DGLEW_STATIC -DGLEW_NO_GLU)

# GLFW (to compile before)
set(GLFW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/glfw-3.3.8")
include_directories(${GLFW_DIR}/include)
link_directories(${GLFW_DIR}/build/src/Release)
SET(GLFW_LIBS glfw3.lib)


######################################################################################################
# BUILD SOLUTION
######################################################################################################

# Add executable for project
add_executable(${PROJECT_NAME} ${SRCS} ${HEADERS} ${PROJECT_SRCS})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBS} ${GLFW_LIBS} ${GLEW_LIBS} ${OPENGL_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75)

# Install executable
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

