cmake_minimum_required(VERSION 3.10)
project(LibtorchExample)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置 Libtorch 的路径
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/pytorch-install/share/cmake/Torch")

# 找到 Libtorch
find_package(Torch REQUIRED)

# 编译 BaseLib 静态库
add_library(BaseLib STATIC src/torch/base.cpp)

# 添加 Libtorch 的头文件路径和编译定义
target_include_directories(BaseLib PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(BaseLib PRIVATE ${TORCH_DEFINITIONS})
target_link_libraries(BaseLib "${TORCH_LIBRARIES}")

# 主程序目标
add_executable(LibtorchExample src/main.cpp)

# 添加 Libtorch 的头文件路径和编译定义给主目标
target_include_directories(LibtorchExample PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(LibtorchExample PRIVATE ${TORCH_DEFINITIONS})
target_link_libraries(LibtorchExample "${TORCH_LIBRARIES}" BaseLib)

# 打印 CUDA 支持状态
if (TORCH_CUDA_ENABLED)
  message(STATUS "CUDA is available. Building with CUDA support.")
else ()
  message(STATUS "CUDA not available. Building without CUDA support.")
endif ()
