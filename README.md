怎么运行？

cd exe
.\collision_sim.exe

可以在1-5选项中选择
1-3.预设模式
4.性能测试
5.手动设置参数模式


# GPU加速碰撞检测系统

## 项目简介
本项目实现了基于GPU的快速大规模碰撞检测算法，使用Spatial Hashing加速技术将碰撞检测复杂度从O(n²)降低到O(n)。

## 系统要求
- Windows 10/11
- NVIDIA GPU (支持CUDA 13.0+)
- Visual Studio 2019/2022
- CMake 3.18+
- Python 3.8+ (用于可视化)
- 依赖：numpy matplotlib ffmpeg

## 编译
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release

# 重新编译
Remove-Item * -Recurse -Force
cmake ..
cmake --build . --config Release

# 绘图
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy matplotlib ffmpeg

cd exe
默认格式
.\collision_sim.exe
性能测试
.\collision_sim.exe --test
选择模式
.collision_sim.exe 2000 600 60 2 -9.8 0.999 0.6 output animation.mp4 1


deactivate