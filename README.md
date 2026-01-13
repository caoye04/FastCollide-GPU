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
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -T cuda=13.1
cmake --build . --config Release
.\Release\collision_sim.exe