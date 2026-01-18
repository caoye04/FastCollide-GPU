cmake_minimum_required(VERSION 3.18)
project(CollisionDetection CUDA CXX)

# 设置C++和CUDA标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 使用更保守的架构设置
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# 查找CUDA
find_package(CUDAToolkit REQUIRED)

# 打印CUDA信息
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# 添加可执行文件
add_executable(collision_sim
    src/main.cu
    src/collision_detection.cu
    src/physics.cu
)

# 设置CUDA编译选项
target_compile_options(collision_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        -Xcompiler=/wd4819
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
)

# 设置输出目录
set_target_properties(collision_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
)