你好！我正在完成我的动画课程的大作业“快速的碰撞检测算法”，下面是我的设备情况和相关作业要求以及我做的一些前期调研。

请你根据已有的所有信息，帮我实现我这次作业的要求！

你可以从整体项目结构、环境配置、代码细节、实验流程等步骤一步一步带我实现！



## 设备情况 

开发环境：VSCODE、Windows、5060ti

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

## 前期调研（粗糙）

### PART1：碰撞检测的加速算法设计

基于对参考资料的研究，本项目拟采用**==空间哈希 Spatial Hashing 算法==**进行碰撞检测加速。这种方法将3D空间划分为均匀网格，通过哈希函数将物体映射到网格单元中，从而将碰撞检测的复杂度从O(n²)降低到O(n)。核心流程如下：

1. **网格划分策略**：根据场景中最大物体的包围盒尺寸确定网格单元大小，确保网格单元边长至少为最大物体直径的1.5倍，再使用3D哈希函数将空间坐标映射到网格ID：

   ```
   hash = (x / cellSize) * prime1 ^ (y / cellSize) * prime2 ^ (z / cellSize) * prime3
   ```

2. **两阶段碰撞检测**：采用**Broad Phase（粗检测）+ Narrow Phase（精检测）**的经典两阶段方法

   - Broad Phase（粗检测阶段）：为每个物体计算其AABB包围盒；确定包围盒覆盖的所有网格单元；将物体ID插入到对应网格单元的列表中；构建潜在碰撞对（同一网格单元内的物体对）
   - Narrow Phase（精检测阶段）：对潜在碰撞对进行精确的几何碰撞测试；检测球心距离是否小于半径之和；计算碰撞响应（位置修正、速度更新）

3. **松散网格优化**：为减少物体在网格边界频繁切换导致的性能损失，采用松散网格（Loose Grid）策略：

   - 定义网格单元的"内边界"和"外边界"
   - 外边界为内边界的2倍大小，允许一定的重叠
   - 物体只有越过外边界时才需要更新所在网格，减少更新频率

### PART2：GPU实现的思路设计

#### 2.1 并行化策略

1. **数据结构设计**：用GPU友好的数据结构，避免动态内存分配

   ```
   // 物体数据
   struct ObjectData {
       float* positions;      // 位置数组 
       float* velocities;     // 速度数组
       float* radii;          // 半径数组
       float* masses;         // 质量数组
       float* restitutions;   // 弹性系数数组
       int count;             // 物体数量
   };
   
   // 网格数据
   struct GridData {
       int* cellStarts;       // 每个网格单元的起始索引
       int* cellEnds;         // 每个网格单元的结束索引
       int* particleIndices;  // 排序后的物体索引
       int* particleHashes;   // 物体对应的网格哈希值
   };
   ```

2.  **CUDA核函数设计**

   - Kernel 1: 计算网格哈希值
     - 每个线程处理一个物体
     - 计算物体AABB包围盒覆盖的网格单元
     - 为每个覆盖的网格单元生成（哈希值，物体ID）对
   - Kernel 2: 基数排序（Radix Sort）
     - 对（哈希值，物体ID）对按哈希值排序
     - 采用GPU优化的并行基数排序算法
     - 分4个pass，每次处理8位（参考NVIDIA GPU Gems 3 Chapter 32）
   - Kernel 3: 构建网格索引
     - 扫描排序后的数组，记录每个网格单元的起始和结束位置
     - 使用并行前缀和（Parallel Prefix Sum）优化
   - Kernel 4: 碰撞检测与响应
     - 每个线程处理一个物体
     - 查询该物体所在网格及相邻26个网格（3D情况）
     - 执行精确碰撞检测和物理响应计算

#### 2.2 内存访问优化

1. **合并内存访问**：使用Structure of Arrays（SoA）布局而非Array of Structures（AoS）；确保相邻线程访问连续内存地址，实现内存访问合并
2. **共享内存利用**：在碰撞检测kernel中，将当前网格的物体数据缓存到共享内存；减少全局内存访问次数
3.  **常量内存**：将场景参数（网格大小、边界等）存储在常量内存中；提高读取效率

#### 2.3 线程组织策略

1. **Block size**: 256线程/块（经验值，平衡占用率和寄存器使用）
2. **Grid size**: 根据物体数量动态计算，确保所有物体被处理
