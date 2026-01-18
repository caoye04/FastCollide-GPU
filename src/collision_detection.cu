#include "collision_detection.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <stdio.h>

// ==================== CUDA核函数 ====================

// 计算3D网格哈希值
__device__ __host__ uint32_t hashPosition(const float3_custom &pos, const SpatialGrid &grid)
{
    int3 gridPos;
    gridPos.x = (int)floorf((pos.x - grid.worldMin.x) / grid.cellSize);
    gridPos.y = (int)floorf((pos.y - grid.worldMin.y) / grid.cellSize);
    gridPos.z = (int)floorf((pos.z - grid.worldMin.z) / grid.cellSize);

    gridPos.x = max(0, min(gridPos.x, grid.gridDim.x - 1));
    gridPos.y = max(0, min(gridPos.y, grid.gridDim.y - 1));
    gridPos.z = max(0, min(gridPos.z, grid.gridDim.z - 1));

    return gridPos.x + gridPos.y * grid.gridDim.x + gridPos.z * grid.gridDim.x * grid.gridDim.y;
}

// Kernel 1: 计算每个粒子的网格哈希值
__global__ void computeHashKernel(
    const float3_custom *positions,
    uint32_t *particleHashes,
    uint32_t *particleIndices,
    int numParticles,
    SpatialGrid grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    particleHashes[idx] = hashPosition(positions[idx], grid);
    particleIndices[idx] = idx;
}

// Kernel 2: 重置网格单元
__global__ void resetGridKernel(
    uint32_t *cellStarts,
    uint32_t *cellEnds,
    uint32_t totalCells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalCells)
        return;

    cellStarts[idx] = 0xFFFFFFFF;
    cellEnds[idx] = 0xFFFFFFFF;
}

// Kernel 3: 构建网格索引
__global__ void buildGridKernel(
    const uint32_t *particleHashes,
    uint32_t *cellStarts,
    uint32_t *cellEnds,
    int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    uint32_t hash = particleHashes[idx];

    if (idx == 0 || hash != particleHashes[idx - 1])
    {
        cellStarts[hash] = idx;
    }

    if (idx == numParticles - 1 || hash != particleHashes[idx + 1])
    {
        cellEnds[hash] = idx + 1;
    }
}

// Kernel 4: 碰撞检测和响应（修复版 - 关键修改）
__global__ void collisionDetectionKernel(
    const float3_custom *positions,
    float3_custom *newPositions,
    const float3_custom *velocities,
    float3_custom *newVelocities,
    const float *radii,
    const float *masses,
    const float *restitutions,
    const uint32_t *particleHashes,
    const uint32_t *particleIndices,
    const uint32_t *cellStarts,
    const uint32_t *cellEnds,
    int *collisionFlags,
    int numParticles,
    SpatialGrid grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

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

    // 累积修正量
    float3_custom totalCorrection(0, 0, 0);
    float3_custom velocityChange(0, 0, 0);
    int collisionCount = 0;

    // 遍历相邻的27个网格单元
    for (int z = -1; z <= 1; z++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                int3 neighborPos = make_int3(
                    gridPos.x + x,
                    gridPos.y + y,
                    gridPos.z + z);

                // 边界检查
                if (neighborPos.x < 0 || neighborPos.x >= grid.gridDim.x ||
                    neighborPos.y < 0 || neighborPos.y >= grid.gridDim.y ||
                    neighborPos.z < 0 || neighborPos.z >= grid.gridDim.z)
                {
                    continue;
                }

                // 计算邻居单元的哈希值
                uint32_t neighborHash = neighborPos.x +
                                        neighborPos.y * grid.gridDim.x +
                                        neighborPos.z * grid.gridDim.x * grid.gridDim.y;

                uint32_t start = cellStarts[neighborHash];
                if (start == 0xFFFFFFFF)
                    continue;

                uint32_t end = cellEnds[neighborHash];

                // 遍历该单元中的所有粒子
                for (uint32_t i = start; i < end; i++)
                {
                    uint32_t otherIdx = particleIndices[i];

                    // ✅ 关键修改：只避免自碰撞，不跳过任何其他粒子
                    if (otherIdx == idx)
                        continue;

                    float3_custom otherPos = positions[otherIdx];
                    float otherRadius = radii[otherIdx];

                    // 计算距离
                    float3_custom delta = pos - otherPos;
                    float dist = delta.length();
                    float minDist = radius + otherRadius;

                    // 碰撞检测
                    if (dist < minDist && dist > 1e-6f)
                    {
                        collisionCount++;

                        // 碰撞响应
                        float3_custom normal = delta.normalize();

                        // ✅ 位置修正 - 每个粒子承担一半的分离距离
                        float overlap = minDist - dist;
                        totalCorrection = totalCorrection + normal * (overlap * 0.5f);

                        // 速度修正（弹性碰撞）
                        float3_custom otherVel = velocities[otherIdx];
                        float otherMass = masses[otherIdx];
                        float otherRestitution = restitutions[otherIdx];

                        float relativeVel = (vel - otherVel).dot(normal);

                        if (relativeVel < 0)
                        {
                            float e = min(restitution, otherRestitution);
                            float j = -(1.0f + e) * relativeVel / (1.0f / mass + 1.0f / otherMass);

                            velocityChange = velocityChange + normal * (j / mass);
                        }
                    }
                }
            }
        }
    }

    // 写入结果
    newPositions[idx] = pos + totalCorrection;
    newVelocities[idx] = vel + velocityChange;
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

CollisionDetector::~CollisionDetector()
{
    freeMemory();
}

void CollisionDetector::allocateMemory()
{
    size_t particleMemory = grid.maxParticles * sizeof(uint32_t) * 2;
    size_t gridMemory = grid.totalCells * sizeof(uint32_t) * 2;
    size_t tempMemory = grid.maxParticles * (sizeof(float3_custom) * 2 + sizeof(int));

    printf("Allocating GPU memory:\n");
    printf("  Particle hashing: %.2f MB\n", particleMemory / 1e6);
    printf("  Grid cells: %.2f MB\n", gridMemory / 1e6);
    printf("  Temporary buffers: %.2f MB\n", tempMemory / 1e6);
    printf("  Total: %.2f MB\n", (particleMemory + gridMemory + tempMemory) / 1e6);

    CUDA_CHECK(cudaMalloc(&grid.particleHashes, grid.maxParticles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&grid.particleIndices, grid.maxParticles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&grid.cellStarts, grid.totalCells * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&grid.cellEnds, grid.totalCells * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_collisionFlags, grid.maxParticles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_newPositions, grid.maxParticles * sizeof(float3_custom)));
    CUDA_CHECK(cudaMalloc(&d_newVelocities, grid.maxParticles * sizeof(float3_custom)));

    printf("GPU memory allocation successful!\n\n");
}

void CollisionDetector::freeMemory()
{
    cudaFree(grid.particleHashes);
    cudaFree(grid.particleIndices);
    cudaFree(grid.cellStarts);
    cudaFree(grid.cellEnds);
    cudaFree(d_collisionFlags);
    cudaFree(d_newPositions);
    cudaFree(d_newVelocities);
}

void CollisionDetector::updateGrid(const ParticleSystem &particles)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int numBlocks = (particles.count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    computeHashKernel<<<numBlocks, BLOCK_SIZE>>>(
        particles.positions,
        grid.particleHashes,
        grid.particleIndices,
        particles.count,
        grid);
    CUDA_CHECK_LAST_ERROR();

    thrust::device_ptr<uint32_t> hashPtr(grid.particleHashes);
    thrust::device_ptr<uint32_t> indexPtr(grid.particleIndices);
    thrust::sort_by_key(hashPtr, hashPtr + particles.count, indexPtr);

    int gridBlocks = (grid.totalCells + BLOCK_SIZE - 1) / BLOCK_SIZE;
    resetGridKernel<<<gridBlocks, BLOCK_SIZE>>>(
        grid.cellStarts,
        grid.cellEnds,
        grid.totalCells);
    CUDA_CHECK_LAST_ERROR();

    buildGridKernel<<<numBlocks, BLOCK_SIZE>>>(
        grid.particleHashes,
        grid.cellStarts,
        grid.cellEnds,
        particles.count);
    CUDA_CHECK_LAST_ERROR();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&stats.gridUpdateTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CollisionDetector::detectAndResolveCollisions(ParticleSystem &particles, float dt)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int numBlocks = (particles.count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    collisionDetectionKernel<<<numBlocks, BLOCK_SIZE>>>(
        particles.positions,
        d_newPositions,
        particles.velocities,
        d_newVelocities,
        particles.radii,
        particles.masses,
        particles.restitutions,
        grid.particleHashes,
        grid.particleIndices,
        grid.cellStarts,
        grid.cellEnds,
        d_collisionFlags,
        particles.count,
        grid);
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaMemcpy(particles.positions, d_newPositions,
                          particles.count * sizeof(float3_custom),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(particles.velocities, d_newVelocities,
                          particles.count * sizeof(float3_custom),
                          cudaMemcpyDeviceToDevice));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&stats.collisionDetectionTime, start, stop);

    // 统计碰撞次数
    int *h_collisionFlags = new int[particles.count];
    CUDA_CHECK(cudaMemcpy(h_collisionFlags, d_collisionFlags,
                          particles.count * sizeof(int), cudaMemcpyDeviceToHost));

    stats.collisionCount = 0;
    for (int i = 0; i < particles.count; i++)
    {
        stats.collisionCount += h_collisionFlags[i];
    }
    // 因为现在每对碰撞被计算两次，所以除以2
    stats.collisionCount /= 2;

    delete[] h_collisionFlags;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}