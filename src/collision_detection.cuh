#ifndef COLLISION_DETECTION_CUH
#define COLLISION_DETECTION_CUH

#include <cuda_runtime.h>
#include <cstdint>

// 常量定义
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_NEIGHBORS = 27; // 3D网格中最多27个相邻单元

// 3D向量结构
struct float3_custom
{
    float x, y, z;

    __host__ __device__ float3_custom() : x(0), y(0), z(0) {}
    __host__ __device__ float3_custom(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ float3_custom operator+(const float3_custom &other) const
    {
        return float3_custom(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ float3_custom operator-(const float3_custom &other) const
    {
        return float3_custom(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ float3_custom operator*(float scalar) const
    {
        return float3_custom(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__ float dot(const float3_custom &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ float length() const
    {
        return sqrtf(x * x + y * y + z * z);
    }

    __host__ __device__ float3_custom normalize() const
    {
        float len = length();
        if (len > 1e-6f)
        {
            return float3_custom(x / len, y / len, z / len);
        }
        return float3_custom(0, 0, 0);
    }
};

// 物体数据结构（SoA布局）
struct ParticleSystem
{
    float3_custom *positions;  // 位置
    float3_custom *velocities; // 速度
    float *radii;              // 半径
    float *masses;             // 质量
    float *restitutions;       // 弹性系数
    int count;                 // 物体数量
};

// 网格数据结构
struct SpatialGrid
{
    uint32_t *particleHashes;  // 每个物体的网格哈希值
    uint32_t *particleIndices; // 排序后的物体索引
    uint32_t *cellStarts;      // 每个网格单元的起始索引
    uint32_t *cellEnds;        // 每个网格单元的结束索引

    float cellSize;         // 网格单元大小
    int3 gridDim;           // 网格维度
    float3_custom worldMin; // 世界空间最小值
    float3_custom worldMax; // 世界空间最大值

    uint32_t maxParticles; // 最大物体数量
    uint32_t totalCells;   // 总网格单元数
};

// 碰撞检测器类
class CollisionDetector
{
public:
    CollisionDetector(int maxParticles, float3_custom worldMin, float3_custom worldMax, float cellSize);
    ~CollisionDetector();

    // 更新空间哈希网格
    void updateGrid(const ParticleSystem &particles);

    // 执行碰撞检测和响应
    void detectAndResolveCollisions(ParticleSystem &particles, float dt);

    // 获取性能统计
    struct Stats
    {
        float gridUpdateTime;
        float collisionDetectionTime;
        int collisionCount;
    };
    Stats getStats() const { return stats; }

private:
    SpatialGrid grid;
    Stats stats;

    // GPU临时缓冲区
    int *d_collisionFlags;

    // 辅助方法
    void allocateMemory();
    void freeMemory();
};

#endif // COLLISION_DETECTION_CUH