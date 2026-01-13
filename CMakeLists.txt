cmake_minimum_required(VERSION 3.18)

# 在 project 之前设置 CUDA 相关变量
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES 89)

project(CollisionDetection LANGUAGES CXX CUDA)

# 设置C++和CUDA标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 显式启用CUDA
enable_language(CUDA)

# 查找CUDA
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(collision_sim
    src/main.cu
    src/collision_detection.cu
    src/physics.cu
)

# 设置CUDA编译选项
set_target_properties(collision_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(collision_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        --generate-line-info
        -arch=sm_89
    >
)

# 链接CUDA库
target_link_libraries(collision_sim PRIVATE
    CUDA::cudart
    CUDA::curand
)

# 设置包含目录
target_include_directories(collision_sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)