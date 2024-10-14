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
	src/mesh.cpp
	src/volume.cpp
	src/main.cpp
	src/interop.cu
    )
    
set(HEADERS
	src/utils.h
    src/comparison.h
    src/basics.h
    src/convolution.h
	src/interop.h
	src/mesh.h
	src/volume.h
	src/gui.h
    )

# Dependencies folder
set(LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libs")
include_directories(SYSTEM "${LIBS_DIR}")


######################################################################################################
# INTERNAL DEPENDENCIES
######################################################################################################


# CUDA ---------------------------------------------------------------
SET(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib/x64)
SET(CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} cudart.lib)
# flag for conditional compilation
add_compile_definitions(USE_CUDA)



# OpenGL -------------------------------------------------------------
find_package(OpenGL REQUIRED)
if(OPENGL_FOUND)
  include_directories(SYSTEM ${OPENGL_INCLUDE_DIR})
endif(OPENGL_FOUND)
# flag for conditional compilation
add_compile_definitions(USE_OPENGL)


# GLEW (download binaries for windows) -------------------------------
set(GLEW_DIR "${LIBS_DIR}/third_party/glew-2.1.0")
include_directories(${GLEW_DIR}/include)
link_directories(${GLEW_DIR}/lib/Release/x64)
# use glew32s only
SET(GLEW_LIBS glew32s.lib)
add_definitions(-DGLEW_STATIC -DGLEW_NO_GLU)


# GLFW (to compile before) -------------------------------------------
set(GLFW_DIR "${LIBS_DIR}/third_party/glfw-3.4")
include_directories(${GLFW_DIR}/include)
link_directories(${GLFW_DIR}/build/src/Release)
SET(GLFW_LIBS glfw3.lib)


# GLM (Header only) --------------------------------------------------
include_directories(SYSTEM "${LIBS_DIR}/third_party/glm-1.0.1")


# ImGui (add sources to project) -------------------------------------
set(IMGUI_DIR "${LIBS_DIR}/third_party/imgui")
aux_source_directory("${IMGUI_DIR}" SRCS)
# new: add appropriate backends
set(IMGUI_BCK "${IMGUI_DIR}/backends/imgui_impl_glfw.h"
			  "${IMGUI_DIR}/backends/imgui_impl_opengl3.h"
			  "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
			  "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp")					
include_directories(SYSTEM "${IMGUI_DIR}")
include_directories(SYSTEM "${IMGUI_DIR}/backends")


# lodepng ------------------------------------------------------------
aux_source_directory("${LIBS_DIR}/third_party/lodepng" SRCS)
include_directories(SYSTEM "${LIBS_DIR}/third_party/lodepng")



######################################################################################################
# BUILD SOLUTION
######################################################################################################


# Add executable for project
add_executable(${PROJECT_NAME} ${SRCS} ${HEADERS} ${IMGUI_BCK})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBS} ${GLFW_LIBS} ${GLEW_LIBS} ${OPENGL_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75)

# Install executable
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

