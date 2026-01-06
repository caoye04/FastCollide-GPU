## PART1：碰撞检测的加速算法设计

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

## PART2：GPU实现的思路设计

### 2.1 并行化策略

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

### 2.2 内存访问优化

1. **合并内存访问**：使用Structure of Arrays（SoA）布局而非Array of Structures（AoS）；确保相邻线程访问连续内存地址，实现内存访问合并
2. **共享内存利用**：在碰撞检测kernel中，将当前网格的物体数据缓存到共享内存；减少全局内存访问次数
3.  **常量内存**：将场景参数（网格大小、边界等）存储在常量内存中；提高读取效率

### 2.3 线程组织策略

1. **Block size**: 256线程/块（经验值，平衡占用率和寄存器使用）
2. **Grid size**: 根据物体数量动态计算，确保所有物体被处理

## PART3：参考文献与网站

1. [[youtube] Spatial Hash Grids & Tales from Game Development](https://www.youtube.com/watch?v=sx4IIQL0x7c)
2. [[zhihu] 空间哈希碰撞检测](https://zhuanlan.zhihu.com/p/480181565)

3. [[tutsplus] Redesign Your Display List With Spatial Hashes](https://code.tutsplus.com/redesign-your-display-list-with-spatial-hashes--cms-27586t)

4. [[CSDN] [宽相检测]空间划分-空间哈希划分均匀网格](https://blog.csdn.net/chenghai37/article/details/153257504)

5. [[CSDN] 并行算法中的哈希技术与空间哈希应用](https://blog.csdn.net/q5r6s7/article/details/151605698?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogOpenSearchComplete%7EPaidSort-1-151605698-blog-153257504.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogOpenSearchComplete%7EPaidSort-1-151605698-blog-153257504.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=1)
6. [[bilibili] Unity空间划分，空间哈希算法](https://www.bilibili.com/video/BV11hZzYMEUg/?spm_id_from=333.337.search-card.all.click&vd_source=94546b19b74a633ecc0cf0e601e9aa7f)
7. [[wiki] Geometric hashing](https://en.wikipedia.org/wiki/Geometric_hashing)
8. [[publication] Mian AS, Bennamoun M, Owens R. Three-dimensional model-based object recognition and segmentation in cluttered scenes. IEEE Trans Pattern Anal Mach Intell. 2006 Oct;28(10):1584-601. doi: 10.1109/TPAMI.2006.213. PMID: 16986541.](https://pubmed.ncbi.nlm.nih.gov/16986541/)
9. [[NVIDIA] GPU Gems 3](https://developer.nvidia.com/gpugems3)
10. [[wiki] AoS and SoA](https://en.wikipedia.org/wiki/AoS_and_SoA) 