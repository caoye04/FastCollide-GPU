你好！我正在完成我的动画课程的大作业“快速的碰撞检测算法”

我已经完成了代码书写，正在试图运行但是似乎遇到了环境上的问题，帮我分析并解决一下

问题如下：

```
PS C:\Users\caoye04\Desktop\FastCollide-GPU\build> cmake ..
CMake Error at C:/Program Files/CMake/share/cmake-4.2/Modules/CMakeDetermineCompilerId.cmake:676 (message):
  No CUDA toolset found.
Call Stack (most recent call first):
  C:/Program Files/CMake/share/cmake-4.2/Modules/CMakeDetermineCompilerId.cmake:8 (CMAKE_DETERMINE_COMPILER_ID_BUILD)
  C:/Program Files/CMake/share/cmake-4.2/Modules/CMakeDetermineCompilerId.cmake:53 (__determine_compiler_id_test)
  C:/Program Files/CMake/share/cmake-4.2/Modules/CMakeDetermineCUDACompiler.cmake:163 (CMAKE_DETERMINE_COMPILER_ID)
  CMakeLists.txt:2 (project)
```



## 设备情况 

开发环境：VSCODE、Windows、5060ti

```
PS C:\Users\caoye04\Desktop\FastCollide-GPU\build> $env:CUDA_PATH
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
PS C:\Users\caoye04\Desktop\FastCollide-GPU\build> where.exe nvcc
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe
PS C:\Users\caoye04\Desktop\FastCollide-GPU\build> & "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" -version
适用于 .NET Framework MSBuild 版本 17.14.23+b0019275e
17.14.23.42201
```

```
C:\Users\caoye04>nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Nov__7_19:25:04_Pacific_Standard_Time_2025
Cuda compilation tools, release 13.1, V13.1.80
Build cuda_13.1.r13.1/compiler.36836380_0
```

```cmd
C:\Users\caoye04>nvidia-smi
Tue Jan  6 21:09:52 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.42                 Driver Version: 581.42         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti   WDDM  |   00000000:01:00.0  On |                  N/A |
|  0%   39C    P0             27W /  180W |    1886MiB /   8151MiB |      4%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2584    C+G   ....0.3650.96\msedgewebview2.exe      N/A      |
|    0   N/A  N/A            5512    C+G   ...em32\ApplicationFrameHost.exe      N/A      |
|    0   N/A  N/A            6452    C+G   ...s\TencentDocs\TencentDocs.exe      N/A      |
|    0   N/A  N/A            7920    C+G   ...y\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A            8876    C+G   ...5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A            9104    C+G   ...acted\runtime\WeChatAppEx.exe      N/A      |
|    0   N/A  N/A            9640    C+G   ..._cw5n1h2txyewy\SearchHost.exe      N/A      |
|    0   N/A  N/A            9856    C+G   ...oda Music\2.5.0\SodaMusic.exe      N/A      |
|    0   N/A  N/A           10200    C+G   ...ffice6\promecefpluginhost.exe      N/A      |
|    0   N/A  N/A           12668    C+G   ...cw5n1h2txyewy\WidgetBoard.exe      N/A      |
|    0   N/A  N/A           13144    C+G   ...t\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A           16236    C+G   ...acted\runtime\WeChatAppEx.exe      N/A      |
|    0   N/A  N/A           16428    C+G   C:\Windows\explorer.exe               N/A      |
|    0   N/A  N/A           16672    C+G   C:\Windows\explorer.exe               N/A      |
|    0   N/A  N/A           16828    C+G   ...kyb3d8bbwe\EdgeGameAssist.exe      N/A      |
|    0   N/A  N/A           18936    C+G   ...les\Tencent\Weixin\Weixin.exe      N/A      |
|    0   N/A  N/A           18988    C+G   ...\cef.win64\steamwebhelper.exe      N/A      |
|    0   N/A  N/A           20380    C+G   ...ms\Microsoft VS Code\Code.exe      N/A      |
|    0   N/A  N/A           20816    C+G   ...ogram Files\Typora\Typora.exe      N/A      |
|    0   N/A  N/A           21440    C+G   ...Chrome\Application\chrome.exe      N/A      |
|    0   N/A  N/A           22904    C+G   ...indows\System32\ShellHost.exe      N/A      |
|    0   N/A  N/A           23936    C+G   ...Chrome\Application\chrome.exe      N/A      |
|    0   N/A  N/A           24632    C+G   ...App_cw5n1h2txyewy\LockApp.exe      N/A      |
|    0   N/A  N/A           25156    C+G   C:\Windows\explorer.exe               N/A      |
|    0   N/A  N/A           25180    C+G   ...8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A           25292    C+G   ...xyewy\ShellExperienceHost.exe      N/A      |
|    0   N/A  N/A           25528    C+G   ...ntrolPanel\SystemSettings.exe      N/A      |
|    0   N/A  N/A           25948    C+G   ...ool\mathpix-snipping-tool.exe      N/A      |
|    0   N/A  N/A           27776    C+G   ...crosoft\OneDrive\OneDrive.exe      N/A      |
|    0   N/A  N/A           28180    C+G   ...__kzh8wxbdkxb8p\DCv2\DCv2.exe      N/A      |
|    0   N/A  N/A           28956    C+G   ...yb3d8bbwe\WindowsTerminal.exe      N/A      |
|    0   N/A  N/A           29532    C+G   C:\Windows\explorer.exe               N/A      |
+-----------------------------------------------------------------------------------------+
```

## 作业要求

### 1. 背景介绍

在动画领域，物体运动和仿真中为了防止穿模，需要进行碰撞检测；在光线追踪渲染中，也要计算光线与物体之间发生的碰撞。所以，一个快速的碰撞检测算法对于大规模运动仿真以及大场景渲染非常重要。  

n个物体的最简单的碰撞检测算法，查询所需的计算复杂度为 O(n2)。可以使用空间划分的数据结构对碰撞检测进行加速，可以将计算复杂度降低为O(n log n)。 如果将算法移植到 GPU 上进行，则可以进行加速。

### 2. 作业内容

作业内容要求如下：
1．实现一种基于 GPU 的快速大规模碰撞检测算法。
2．测试分析算法的性能。
3．将算法应用在如下的应用中：一个固定场景中有大量小球或者物体。各个小球或物体有不同的半径、质量、初速度和弹性系数，利用所实现的最近邻查找算法对小球或物体的运动和碰撞进行仿真，制作一段动画。其中，小球或物体均作为刚体考虑即可，无需考虑碰撞引起的自身形变。动画制作可以自己完成，也可以使用现有的软件进行渲染。

### 3. 评分要求

1. 运行效果（功能、效率、bug） $40 \%$
   （a）程序应当正确仿真物体的运动及其动画效果 $10 \%$
   （b）程序应当正确仿真物体的碰撞及其动画效果 $10 \%$
   （c）程序应当能够达到应有的效率 $20 \%$
2. 代码质量 $30 \%$
   （a）代码应当包含必要注释 $10 \%$
   （b）代码的风格应当具有统一性 $10 \%$
   （c）代码应当具有可移植性 $10 \%$
3. 中期文档（第 9 周提交） $5 \%$
   （a）说明碰撞检测的加速算法设计 $2 \%$
   （b）说明GPU实现的思路设计 $2 \%$
   （c）参考文献 $1 \%$
4. 结题文档（第 18 周提交） $10 \%$
   （a）说明程序运行环境，以及项目和代码依赖的编程环境 $1 \%$
   （b）各个程序模块之间的逻辑关系 $1 \%$
   （c）简要说明各个功能的演示方法 $1 \%$
   （d）程序运行的主要流程 $2 \%$
   （e）算法性能测试结果分析 $4 \%$
   （f）参考文献或引用代码出处 $1 \%$

## 我的实现

### 代码框架

```
CollisionDetection/
├── src/
│   ├── main.cu
│   ├── collision_detection.cuh
│   ├── collision_detection.cu
│   ├── physics.cuh
│   └── physics.cu
├── output/
│   └── (动画输出目录)
├── CMakeLists.txt
└── README.md
```

### CMakeLists.txt

```
cmake_minimum_required(VERSION 3.18)
project(CollisionDetection CUDA CXX)

# 设置C++和CUDA标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 5060Ti架构

# 查找CUDA
find_package(CUDAToolkit REQUIRED)

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
        --generate-line-info
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
```

### **src/collision_detection.cuh:**

```
#ifndef COLLISION_DETECTION_CUH
#define COLLISION_DETECTION_CUH

#include <cuda_runtime.h>
#include <cstdint>

// 常量定义
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_NEIGHBORS = 27;  // 3D网格中最多27个相邻单元

// 3D向量结构
struct float3_custom {
    float x, y, z;
    
    __host__ __device__ float3_custom() : x(0), y(0), z(0) {}
    __host__ __device__ float3_custom(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    __host__ __device__ float3_custom operator+(const float3_custom& other) const {
        return float3_custom(x + other.x, y + other.y, z + other.z);
    }
    
    __host__ __device__ float3_custom operator-(const float3_custom& other) const {
        return float3_custom(x - other.x, y - other.y, z - other.z);
    }
    
    __host__ __device__ float3_custom operator*(float scalar) const {
        return float3_custom(x * scalar, y * scalar, z * scalar);
    }
    
    __host__ __device__ float dot(const float3_custom& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }
    
    __host__ __device__ float3_custom normalize() const {
        float len = length();
        if (len > 1e-6f) {
            return float3_custom(x / len, y / len, z / len);
        }
        return float3_custom(0, 0, 0);
    }
};

// 物体数据结构（SoA布局）
struct ParticleSystem {
    float3_custom* positions;      // 位置
    float3_custom* velocities;     // 速度
    float* radii;                  // 半径
    float* masses;                 // 质量
    float* restitutions;           // 弹性系数
    int count;                     // 物体数量
};

// 网格数据结构
struct SpatialGrid {
    uint32_t* particleHashes;      // 每个物体的网格哈希值
    uint32_t* particleIndices;     // 排序后的物体索引
    uint32_t* cellStarts;          // 每个网格单元的起始索引
    uint32_t* cellEnds;            // 每个网格单元的结束索引
    
    float cellSize;                // 网格单元大小
    int3 gridDim;                  // 网格维度
    float3_custom worldMin;        // 世界空间最小值
    float3_custom worldMax;        // 世界空间最大值
    
    uint32_t maxParticles;         // 最大物体数量
    uint32_t totalCells;           // 总网格单元数
};

// 碰撞检测器类
class CollisionDetector {
public:
    CollisionDetector(int maxParticles, float3_custom worldMin, float3_custom worldMax, float cellSize);
    ~CollisionDetector();
    
    // 更新空间哈希网格
    void updateGrid(const ParticleSystem& particles);
    
    // 执行碰撞检测和响应
    void detectAndResolveCollisions(ParticleSystem& particles, float dt);
    
    // 获取性能统计
    struct Stats {
        float gridUpdateTime;
        float collisionDetectionTime;
        int collisionCount;
    };
    Stats getStats() const { return stats; }
    
private:
    SpatialGrid grid;
    Stats stats;
    
    // GPU临时缓冲区
    int* d_collisionFlags;
    
    // 辅助方法
    void allocateMemory();
    void freeMemory();
};

#endif // COLLISION_DETECTION_CUH
```

### src/physics.cuh:

```
#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#include "collision_detection.cuh"
#include <curand_kernel.h>

// 物理仿真参数
struct SimulationParams {
    float3_custom gravity;         // 重力加速度
    float3_custom worldMin;        // 世界边界
    float3_custom worldMax;
    float damping;                 // 阻尼系数
    float groundRestitution;       // 地面弹性系数
};

// 物理仿真器类
class PhysicsSimulator {
public:
    PhysicsSimulator(const SimulationParams& params);
    ~PhysicsSimulator();
    
    // 初始化粒子系统
    void initializeParticles(ParticleSystem& particles, int count);
    
    // 积分更新位置和速度
    void integrate(ParticleSystem& particles, float dt);
    
    // 处理边界碰撞
    void handleBoundaryCollisions(ParticleSystem& particles);
    
    // 获取仿真参数
    const SimulationParams& getParams() const { return params; }
    
private:
    SimulationParams params;
    curandState* d_randStates;
    
    void initRandom(int count);
};

#endif // PHYSICS_CUH
```

### src/collision_detection.cu:

```
#include "collision_detection.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <stdio.h>

// ==================== CUDA核函数 ====================

// 计算3D网格哈希值
__device__ __host__ uint32_t hashPosition(const float3_custom& pos, const SpatialGrid& grid) {
    int3 gridPos;
    gridPos.x = (int)floorf((pos.x - grid.worldMin.x) / grid.cellSize);
    gridPos.y = (int)floorf((pos.y - grid.worldMin.y) / grid.cellSize);
    gridPos.z = (int)floorf((pos.z - grid.worldMin.z) / grid.cellSize);
    
    // 边界检查
    gridPos.x = max(0, min(gridPos.x, grid.gridDim.x - 1));
    gridPos.y = max(0, min(gridPos.y, grid.gridDim.y - 1));
    gridPos.z = max(0, min(gridPos.z, grid.gridDim.z - 1));
    
    // 计算哈希值
    return gridPos.x + gridPos.y * grid.gridDim.x + gridPos.z * grid.gridDim.x * grid.gridDim.y;
}

// Kernel 1: 计算每个粒子的网格哈希值
__global__ void computeHashKernel(
    const float3_custom* positions,
    uint32_t* particleHashes,
    uint32_t* particleIndices,
    int numParticles,
    SpatialGrid grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    // 计算哈希值
    particleHashes[idx] = hashPosition(positions[idx], grid);
    particleIndices[idx] = idx;
}

// Kernel 2: 重置网格单元
__global__ void resetGridKernel(
    uint32_t* cellStarts,
    uint32_t* cellEnds,
    uint32_t totalCells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalCells) return;
    
    cellStarts[idx] = 0xFFFFFFFF;
    cellEnds[idx] = 0xFFFFFFFF;
}

// Kernel 3: 构建网格索引
__global__ void buildGridKernel(
    const uint32_t* particleHashes,
    uint32_t* cellStarts,
    uint32_t* cellEnds,
    int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    uint32_t hash = particleHashes[idx];
    
    // 如果是该单元的第一个粒子
    if (idx == 0 || hash != particleHashes[idx - 1]) {
        cellStarts[hash] = idx;
    }
    
    // 如果是该单元的最后一个粒子
    if (idx == numParticles - 1 || hash != particleHashes[idx + 1]) {
        cellEnds[hash] = idx + 1;
    }
}

// Kernel 4: 碰撞检测和响应
__global__ void collisionDetectionKernel(
    float3_custom* positions,
    float3_custom* velocities,
    const float* radii,
    const float* masses,
    const float* restitutions,
    const uint32_t* particleHashes,
    const uint32_t* particleIndices,
    const uint32_t* cellStarts,
    const uint32_t* cellEnds,
    int* collisionFlags,
    int numParticles,
    SpatialGrid grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    // 获取当前粒子数据
    float3_custom pos = positions[idx];
    float3_custom vel = velocities[idx];
    float radius = radii[idx];
    float mass = masses[idx];
    float restitution = restitutions[idx];
    
    // 计算当前粒子所在的网格位置
    int3 gridPos;
    gridPos.x = (int)floorf((pos.x - grid.worldMin.x) / grid.cellSize);
    gridPos.y = (int)floorf((pos.y - grid.worldMin.y) / grid.cellSize);
    gridPos.z = (int)floorf((pos.z - grid.worldMin.z) / grid.cellSize);
    
    int collisionCount = 0;
    
    // 遍历相邻的27个网格单元
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighborPos = make_int3(
                    gridPos.x + x,
                    gridPos.y + y,
                    gridPos.z + z
                );
                
                // 边界检查
                if (neighborPos.x < 0 || neighborPos.x >= grid.gridDim.x ||
                    neighborPos.y < 0 || neighborPos.y >= grid.gridDim.y ||
                    neighborPos.z < 0 || neighborPos.z >= grid.gridDim.z) {
                    continue;
                }
                
                // 计算邻居单元的哈希值
                uint32_t neighborHash = neighborPos.x + 
                                       neighborPos.y * grid.gridDim.x + 
                                       neighborPos.z * grid.gridDim.x * grid.gridDim.y;
                
                uint32_t start = cellStarts[neighborHash];
                if (start == 0xFFFFFFFF) continue;
                
                uint32_t end = cellEnds[neighborHash];
                
                // 遍历该单元中的所有粒子
                for (uint32_t i = start; i < end; i++) {
                    uint32_t otherIdx = particleIndices[i];
                    
                    // 避免自碰撞和重复检测
                    if (otherIdx <= idx) continue;
                    
                    float3_custom otherPos = positions[otherIdx];
                    float otherRadius = radii[otherIdx];
                    
                    // 计算距离
                    float3_custom delta = pos - otherPos;
                    float dist = delta.length();
                    float minDist = radius + otherRadius;
                    
                    // 碰撞检测
                    if (dist < minDist && dist > 1e-6f) {
                        collisionCount++;
                        
                        // 碰撞响应
                        float3_custom normal = delta.normalize();
                        
                        // 位置修正（分离物体）
                        float overlap = minDist - dist;
                        float3_custom correction = normal * (overlap * 0.5f);
                        
                        positions[idx] = pos + correction;
                        positions[otherIdx] = otherPos - correction;
                        
                        // 速度修正（弹性碰撞）
                        float3_custom otherVel = velocities[otherIdx];
                        float otherMass = masses[otherIdx];
                        float otherRestitution = restitutions[otherIdx];
                        
                        float relativeVel = (vel - otherVel).dot(normal);
                        
                        if (relativeVel < 0) {  // 只处理接近的碰撞
                            float e = min(restitution, otherRestitution);  // 取较小的弹性系数
                            float j = -(1.0f + e) * relativeVel / (1.0f / mass + 1.0f / otherMass);
                            
                            float3_custom impulse = normal * j;
                            
                            velocities[idx] = vel + impulse * (1.0f / mass);
                            velocities[otherIdx] = otherVel - impulse * (1.0f / otherMass);
                        }
                    }
                }
            }
        }
    }
    
    collisionFlags[idx] = collisionCount;
}

// ==================== CollisionDetector实现 ====================

CollisionDetector::CollisionDetector(
    int maxParticles, 
    float3_custom worldMin, 
    float3_custom worldMax, 
    float cellSize)
{
    grid.maxParticles = maxParticles;
    grid.cellSize = cellSize;
    grid.worldMin = worldMin;
    grid.worldMax = worldMax;
    
    // 计算网格维度
    float3_custom worldSize = worldMax - worldMin;
    grid.gridDim.x = (int)ceilf(worldSize.x / cellSize);
    grid.gridDim.y = (int)ceilf(worldSize.y / cellSize);
    grid.gridDim.z = (int)ceilf(worldSize.z / cellSize);
    grid.totalCells = grid.gridDim.x * grid.gridDim.y * grid.gridDim.z;
    
    printf("Grid dimensions: %d x %d x %d = %d cells\n", 
           grid.gridDim.x, grid.gridDim.y, grid.gridDim.z, grid.totalCells);
    
    allocateMemory();
    
    stats = {0, 0, 0};
}

CollisionDetector::~CollisionDetector() {
    freeMemory();
}

void CollisionDetector::allocateMemory() {
    cudaMalloc(&grid.particleHashes, grid.maxParticles * sizeof(uint32_t));
    cudaMalloc(&grid.particleIndices, grid.maxParticles * sizeof(uint32_t));
    cudaMalloc(&grid.cellStarts, grid.totalCells * sizeof(uint32_t));
    cudaMalloc(&grid.cellEnds, grid.totalCells * sizeof(uint32_t));
    cudaMalloc(&d_collisionFlags, grid.maxParticles * sizeof(int));
}

void CollisionDetector::freeMemory() {
    cudaFree(grid.particleHashes);
    cudaFree(grid.particleIndices);
    cudaFree(grid.cellStarts);
    cudaFree(grid.cellEnds);
    cudaFree(d_collisionFlags);
}

void CollisionDetector::updateGrid(const ParticleSystem& particles) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int numBlocks = (particles.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 步骤1: 计算哈希值
    computeHashKernel<<<numBlocks, BLOCK_SIZE>>>(
        particles.positions,
        grid.particleHashes,
        grid.particleIndices,
        particles.count,
        grid
    );
    
    // 步骤2: 排序
    thrust::device_ptr<uint32_t> hashPtr(grid.particleHashes);
    thrust::device_ptr<uint32_t> indexPtr(grid.particleIndices);
    thrust::sort_by_key(hashPtr, hashPtr + particles.count, indexPtr);
    
    // 步骤3: 重置网格
    int gridBlocks = (grid.totalCells + BLOCK_SIZE - 1) / BLOCK_SIZE;
    resetGridKernel<<<gridBlocks, BLOCK_SIZE>>>(
        grid.cellStarts,
        grid.cellEnds,
        grid.totalCells
    );
    
    // 步骤4: 构建网格索引
    buildGridKernel<<<numBlocks, BLOCK_SIZE>>>(
        grid.particleHashes,
        grid.cellStarts,
        grid.cellEnds,
        particles.count
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&stats.gridUpdateTime, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CollisionDetector::detectAndResolveCollisions(ParticleSystem& particles, float dt) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int numBlocks = (particles.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 执行碰撞检测和响应
    collisionDetectionKernel<<<numBlocks, BLOCK_SIZE>>>(
        particles.positions,
        particles.velocities,
        particles.radii,
        particles.masses,
        particles.restitutions,
        grid.particleHashes,
        grid.particleIndices,
        grid.cellStarts,
        grid.cellEnds,
        d_collisionFlags,
        particles.count,
        grid
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&stats.collisionDetectionTime, start, stop);
    
    // 统计碰撞次数
    int* h_collisionFlags = new int[particles.count];
    cudaMemcpy(h_collisionFlags, d_collisionFlags, particles.count * sizeof(int), cudaMemcpyDeviceToHost);
    
    stats.collisionCount = 0;
    for (int i = 0; i < particles.count; i++) {
        stats.collisionCount += h_collisionFlags[i];
    }
    
    delete[] h_collisionFlags;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

### src/physics.cu:

```
#include "physics.cuh"
#include <curand_kernel.h>
#include <stdio.h>

// ==================== CUDA核函数 ====================

// 初始化随机数生成器
__global__ void initRandKernel(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    curand_init(seed, idx, 0, &states[idx]);
}

// 随机初始化粒子
__global__ void initParticlesKernel(
    float3_custom* positions,
    float3_custom* velocities,
    float* radii,
    float* masses,
    float* restitutions,
    curandState* randStates,
    int count,
    float3_custom worldMin,
    float3_custom worldMax)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    curandState localState = randStates[idx];
    
    // 随机位置
    float x = worldMin.x + curand_uniform(&localState) * (worldMax.x - worldMin.x);
    float y = worldMin.y + 0.3f * (worldMax.y - worldMin.y) + 
              curand_uniform(&localState) * 0.6f * (worldMax.y - worldMin.y);
    float z = worldMin.z + curand_uniform(&localState) * (worldMax.z - worldMin.z);
    positions[idx] = float3_custom(x, y, z);
    
    // 随机速度
    float vx = (curand_uniform(&localState) - 0.5f) * 5.0f;
    float vy = curand_uniform(&localState) * 2.0f;
    float vz = (curand_uniform(&localState) - 0.5f) * 5.0f;
    velocities[idx] = float3_custom(vx, vy, vz);
    
    // 随机半径 (0.1 到 0.3)
    radii[idx] = 0.1f + curand_uniform(&localState) * 0.2f;
    
    // 质量与半径立方成正比
    float r = radii[idx];
    masses[idx] = 4.0f / 3.0f * 3.14159f * r * r * r * 1000.0f;  // 密度1000
    
    // 随机弹性系数 (0.5 到 0.95)
    restitutions[idx] = 0.5f + curand_uniform(&localState) * 0.45f;
    
    randStates[idx] = localState;
}

// 积分更新（Velocity Verlet）
__global__ void integrateKernel(
    float3_custom* positions,
    float3_custom* velocities,
    const float* masses,
    int count,
    float3_custom gravity,
    float dt,
    float damping)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    float3_custom pos = positions[idx];
    float3_custom vel = velocities[idx];
    
    // 应用重力
    vel = vel + gravity * dt;
    
    // 应用阻尼
    vel = vel * damping;
    
    // 更新位置
    pos = pos + vel * dt;
    
    positions[idx] = pos;
    velocities[idx] = vel;
}

// 处理边界碰撞
__global__ void boundaryCollisionKernel(
    float3_custom* positions,
    float3_custom* velocities,
    const float* radii,
    const float* restitutions,
    int count,
    float3_custom worldMin,
    float3_custom worldMax,
    float groundRestitution)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    float3_custom pos = positions[idx];
    float3_custom vel = velocities[idx];
    float radius = radii[idx];
    float restitution = restitutions[idx];
    
    // X边界
    if (pos.x - radius < worldMin.x) {
        pos.x = worldMin.x + radius;
        vel.x = -vel.x * restitution;
    }
    if (pos.x + radius > worldMax.x) {
        pos.x = worldMax.x - radius;
        vel.x = -vel.x * restitution;
    }
    
    // Y边界（地面使用特殊弹性系数）
    if (pos.y - radius < worldMin.y) {
        pos.y = worldMin.y + radius;
        vel.y = -vel.y * groundRestitution;
        vel.x *= 0.98f;  // 地面摩擦
        vel.z *= 0.98f;
    }
    if (pos.y + radius > worldMax.y) {
        pos.y = worldMax.y - radius;
        vel.y = -vel.y * restitution;
    }
    
    // Z边界
    if (pos.z - radius < worldMin.z) {
        pos.z = worldMin.z + radius;
        vel.z = -vel.z * restitution;
    }
    if (pos.z + radius > worldMax.z) {
        pos.z = worldMax.z - radius;
        vel.z = -vel.z * restitution;
    }
    
    positions[idx] = pos;
    velocities[idx] = vel;
}

// ==================== PhysicsSimulator实现 ====================

PhysicsSimulator::PhysicsSimulator(const SimulationParams& params) : params(params) {
    d_randStates = nullptr;
}

PhysicsSimulator::~PhysicsSimulator() {
    if (d_randStates) {
        cudaFree(d_randStates);
    }
}

void PhysicsSimulator::initRandom(int count) {
    cudaMalloc(&d_randStates, count * sizeof(curandState));
    
    int numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initRandKernel<<<numBlocks, BLOCK_SIZE>>>(d_randStates, time(nullptr), count);
    cudaDeviceSynchronize();
}

void PhysicsSimulator::initializeParticles(ParticleSystem& particles, int count) {
    particles.count = count;
    
    // 分配内存
    cudaMalloc(&particles.positions, count * sizeof(float3_custom));
    cudaMalloc(&particles.velocities, count * sizeof(float3_custom));
    cudaMalloc(&particles.radii, count * sizeof(float));
    cudaMalloc(&particles.masses, count * sizeof(float));
    cudaMalloc(&particles.restitutions, count * sizeof(float));
    
    // 初始化随机数生成器
    initRandom(count);
    
    // 初始化粒子
    int numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initParticlesKernel<<<numBlocks, BLOCK_SIZE>>>(
        particles.positions,
        particles.velocities,
        particles.radii,
        particles.masses,
        particles.restitutions,
        d_randStates,
        count,
        params.worldMin,
        params.worldMax
    );
    
    cudaDeviceSynchronize();
    printf("Initialized %d particles\n", count);
}

void PhysicsSimulator::integrate(ParticleSystem& particles, float dt) {
    int numBlocks = (particles.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    integrateKernel<<<numBlocks, BLOCK_SIZE>>>(
        particles.positions,
        particles.velocities,
        particles.masses,
        particles.count,
        params.gravity,
        dt,
        params.damping
    );
}

void PhysicsSimulator::handleBoundaryCollisions(ParticleSystem& particles) {
    int numBlocks = (particles.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    boundaryCollisionKernel<<<numBlocks, BLOCK_SIZE>>>(
        particles.positions,
        particles.velocities,
        particles.radii,
        particles.restitutions,
        particles.count,
        params.worldMin,
        params.worldMax,
        params.groundRestitution
    );
}
```

### main.cu

```
#include "collision_detection.cuh"
#include "physics.cuh"
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

// 导出数据到文件
void exportFrame(const ParticleSystem& particles, int frameNum, const std::string& outputDir) {
    // 从GPU复制数据到CPU
    std::vector<float3_custom> positions(particles.count);
    std::vector<float> radii(particles.count);
    
    cudaMemcpy(positions.data(), particles.positions, 
               particles.count * sizeof(float3_custom), cudaMemcpyDeviceToHost);
    cudaMemcpy(radii.data(), particles.radii, 
               particles.count * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 写入文件
    char filename[256];
    sprintf(filename, "%s/frame_%04d.txt", outputDir.c_str(), frameNum);
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("Failed to open file: %s\n", filename);
        return;
    }
    
    fprintf(fp, "%d\n", particles.count);
    for (int i = 0; i < particles.count; i++) {
        fprintf(fp, "%.6f %.6f %.6f %.6f\n", 
                positions[i].x, positions[i].y, positions[i].z, radii[i]);
    }
    
    fclose(fp);
}

// 性能测试
void performanceTest() {
    printf("\n========== Performance Test ==========\n");
    
    int testCounts[] = {1000, 5000, 10000, 20000, 50000};
    
    SimulationParams params;
    params.gravity = float3_custom(0, -9.8f, 0);
    params.worldMin = float3_custom(-10, 0, -10);
    params.worldMax = float3_custom(10, 20, 10);
    params.damping = 0.999f;
    params.groundRestitution = 0.6f;
    
    for (int count : testCounts) {
        printf("\nTesting with %d particles...\n", count);
        
        // 创建仿真器和碰撞检测器
        PhysicsSimulator simulator(params);
        CollisionDetector detector(count * 2, params.worldMin, params.worldMax, 0.6f);
        
        // 初始化粒子
        ParticleSystem particles;
        simulator.initializeParticles(particles, count);
        
        // 预热
        for (int i = 0; i < 10; i++) {
            detector.updateGrid(particles);
            detector.detectAndResolveCollisions(particles, 0.016f);
        }
        
        // 测试100帧
        auto startTime = std::chrono::high_resolution_clock::now();
        
        float totalGridTime = 0;
        float totalCollisionTime = 0;
        int totalCollisions = 0;
        
        for (int frame = 0; frame < 100; frame++) {
            simulator.integrate(particles, 0.016f);
            simulator.handleBoundaryCollisions(particles);
            
            detector.updateGrid(particles);
            detector.detectAndResolveCollisions(particles, 0.016f);
            
            auto stats = detector.getStats();
            totalGridTime += stats.gridUpdateTime;
            totalCollisionTime += stats.collisionDetectionTime;
            totalCollisions += stats.collisionCount;
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        
        printf("  Total time: %.2f ms (%.2f FPS)\n", totalTime, 100000.0f / totalTime);
        printf("  Avg grid update: %.3f ms\n", totalGridTime / 100);
        printf("  Avg collision detection: %.3f ms\n", totalCollisionTime / 100);
        printf("  Avg collisions per frame: %d\n", totalCollisions / 100);
        
        // 清理
        cudaFree(particles.positions);
        cudaFree(particles.velocities);
        cudaFree(particles.radii);
        cudaFree(particles.masses);
        cudaFree(particles.restitutions);
    }
}

int main(int argc, char** argv) {
    printf("GPU Collision Detection System\n");
    printf("==============================\n\n");
    
    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n\n", prop.totalGlobalMem / 1e9);
    
    // 运行性能测试
    if (argc > 1 && std::string(argv[1]) == "--test") {
        performanceTest();
        return 0;
    }
    
    // 仿真参数
    SimulationParams params;
    params.gravity = float3_custom(0, -9.8f, 0);
    params.worldMin = float3_custom(-10, 0, -10);
    params.worldMax = float3_custom(10, 20, 10);
    params.damping = 0.999f;
    params.groundRestitution = 0.6f;
    
    // 创建仿真器和碰撞检测器
    int numParticles = 5000;
    PhysicsSimulator simulator(params);
    CollisionDetector detector(numParticles * 2, params.worldMin, params.worldMax, 0.6f);
    
    // 初始化粒子系统
    ParticleSystem particles;
    simulator.initializeParticles(particles, numParticles);
    
    // 创建输出目录
    system("mkdir output 2>nul");
    
    // 仿真循环
    float dt = 1.0f / 60.0f;  // 60 FPS
    int totalFrames = 600;    // 10秒动画
    int exportInterval = 2;   // 每2帧导出一次
    
    printf("Starting simulation...\n");
    printf("Simulating %d frames at %.1f FPS\n", totalFrames, 1.0f / dt);
    printf("Output directory: ./output/\n\n");
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int frame = 0; frame < totalFrames; frame++) {
        // 物理积分
        simulator.integrate(particles, dt);
        
        // 边界碰撞
        simulator.handleBoundaryCollisions(particles);
        
        // 更新空间网格
        detector.updateGrid(particles);
        
        // 碰撞检测和响应
        detector.detectAndResolveCollisions(particles, dt);
        
        // 导出帧数据
        if (frame % exportInterval == 0) {
            exportFrame(particles, frame / exportInterval, "output");
        }
        
        // 显示进度
        if (frame % 60 == 0) {
            auto stats = detector.getStats();
            printf("Frame %d/%d - Grid: %.2fms, Collision: %.2fms, Collisions: %d\n",
                   frame, totalFrames, stats.gridUpdateTime, 
                   stats.collisionDetectionTime, stats.collisionCount);
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    float totalTime = std::chrono::duration<float>(endTime - startTime).count();
    
    printf("\nSimulation complete!\n");
    printf("Total time: %.2f seconds (%.2f FPS)\n", totalTime, totalFrames / totalTime);
    printf("Frames exported: %d\n", totalFrames / exportInterval);
    
    // 清理
    cudaFree(particles.positions);
    cudaFree(particles.velocities);
    cudaFree(particles.radii);
    cudaFree(particles.masses);
    cudaFree(particles.restitutions);
    
    return 0;
}
```

### visualize.py

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import glob
import os

def load_frame(filename):
    """加载单帧数据"""
    with open(filename, 'r') as f:
        num_particles = int(f.readline())
        data = np.loadtxt(f)
    return data

def create_animation(output_dir='output', output_file='animation.mp4'):
    """创建3D动画"""
    # 获取所有帧文件
    frame_files = sorted(glob.glob(os.path.join(output_dir, 'frame_*.txt')))
    
    if not frame_files:
        print(f"No frame files found in {output_dir}")
        return
    
    print(f"Found {len(frame_files)} frames")
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 20)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('GPU Collision Detection Simulation')
    
    # 初始化散点图
    data = load_frame(frame_files[0])
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                        s=data[:, 3] * 1000, c=data[:, 3], 
                        cmap='viridis', alpha=0.6)
    
    def update(frame_idx):
        """更新函数"""
        data = load_frame(frame_files[frame_idx])
        scatter._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
        scatter.set_sizes(data[:, 3] * 1000)
        ax.set_title(f'Frame {frame_idx}/{len(frame_files)}')
        return scatter,
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(frame_files),
                        interval=33, blit=False)
    
    # 保存动画
    print(f"Saving animation to {output_file}...")
    writer = FFMpegWriter(fps=30, bitrate=5000)
    anim.save(output_file, writer=writer)
    print("Animation saved!")

if __name__ == '__main__':
    create_animation()
```

### readme.md

```
# GPU加速碰撞检测系统

## 项目简介
本项目实现了基于GPU的快速大规模碰撞检测算法，使用Spatial Hashing加速技术将碰撞检测复杂度从O(n²)降低到O(n)。

## 系统要求
- Windows 10/11
- NVIDIA GPU (支持CUDA 13.0+)
- Visual Studio 2019/2022
- CMake 3.18+
- Python 3.8+ (用于可视化)

## 编译运行
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
.\Release\collision_sim.exe
```

