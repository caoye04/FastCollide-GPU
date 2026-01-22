#include "physics.cuh"
#include <curand_kernel.h>
#include <stdio.h>

// 初始化随机数生成器
__global__ void initRandKernel(curandState *states, unsigned long seed, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    curand_init(seed, idx, 0, &states[idx]);
}

// 随机初始化粒子 - 扩大参数差异范围
__global__ void initParticlesKernel(
    float3_custom *positions,
    float3_custom *velocities,
    float *radii,
    float *masses,
    float *restitutions,
    curandState *randStates,
    int count,
    float3_custom worldMin,
    float3_custom worldMax)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    curandState localState = randStates[idx];

    // 随机位置（Z轴为高度，分布更广）
    float x = worldMin.x + curand_uniform(&localState) * (worldMax.x - worldMin.x);
    float y = worldMin.y + curand_uniform(&localState) * (worldMax.y - worldMin.y);
    float z = worldMin.z + 0.2f * (worldMax.z - worldMin.z) +
              curand_uniform(&localState) * 0.7f * (worldMax.z - worldMin.z);
    positions[idx] = float3_custom(x, y, z);

    // ✅ 更大范围的随机速度（-8 到 +8）
    float vx = (curand_uniform(&localState) - 0.5f) * 16.0f;
    float vy = (curand_uniform(&localState) - 0.5f) * 16.0f;
    float vz = curand_uniform(&localState) * 8.0f; // Z方向向上的初速度
    velocities[idx] = float3_custom(vx, vy, vz);

    // ✅ 更大范围的随机半径 (0.05 到 0.4)
    radii[idx] = 0.05f + curand_uniform(&localState) * 0.35f;

    // ✅ 质量与半径立方成正比，但密度也随机变化（500-1500 kg/m³）
    float r = radii[idx];
    float density = 500.0f + curand_uniform(&localState) * 1000.0f;
    masses[idx] = 4.0f / 3.0f * 3.14159f * r * r * r * density;

    // ✅ 更大范围的随机弹性系数 (0.3 到 0.95)
    restitutions[idx] = 0.3f + curand_uniform(&localState) * 0.65f;

    randStates[idx] = localState;
}

// 积分更新（Velocity Verlet）
__global__ void integrateKernel(
    float3_custom *positions,
    float3_custom *velocities,
    const float *masses,
    int count,
    float3_custom gravity,
    float dt,
    float damping)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

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
    float3_custom *positions,
    float3_custom *velocities,
    const float *radii,
    const float *restitutions,
    int count,
    float3_custom worldMin,
    float3_custom worldMax,
    float groundRestitution)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    float3_custom pos = positions[idx];
    float3_custom vel = velocities[idx];
    float radius = radii[idx];
    float restitution = restitutions[idx];

    // X边界
    if (pos.x - radius < worldMin.x)
    {
        pos.x = worldMin.x + radius;
        vel.x = -vel.x * restitution;
    }
    if (pos.x + radius > worldMax.x)
    {
        pos.x = worldMax.x - radius;
        vel.x = -vel.x * restitution;
    }

    // Y边界
    if (pos.y - radius < worldMin.y)
    {
        pos.y = worldMin.y + radius;
        vel.y = -vel.y * restitution;
    }
    if (pos.y + radius > worldMax.y)
    {
        pos.y = worldMax.y - radius;
        vel.y = -vel.y * restitution;
    }

    // Z边界（地面和天花板）
    if (pos.z - radius < worldMin.z)
    {
        pos.z = worldMin.z + radius;
        vel.z = -vel.z * groundRestitution;
        vel.x *= 0.95f; // 地面摩擦
        vel.y *= 0.95f;
    }
    if (pos.z + radius > worldMax.z)
    {
        pos.z = worldMax.z - radius;
        vel.z = -vel.z * restitution;
    }

    positions[idx] = pos;
    velocities[idx] = vel;
}

// ==================== PhysicsSimulator实现 ====================

PhysicsSimulator::PhysicsSimulator(const SimulationParams &params) : params(params)
{
    d_randStates = nullptr;
}

PhysicsSimulator::~PhysicsSimulator()
{
    if (d_randStates)
    {
        cudaFree(d_randStates);
    }
}

void PhysicsSimulator::initRandom(int count)
{
    CUDA_CHECK(cudaMalloc(&d_randStates, count * sizeof(curandState)));

    int numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initRandKernel<<<numBlocks, BLOCK_SIZE>>>(d_randStates, time(nullptr), count);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void PhysicsSimulator::initializeParticles(ParticleSystem &particles, int count)
{
    particles.count = count;

    printf("Allocating particle system memory for %d particles...\n", count);

    CUDA_CHECK(cudaMalloc(&particles.positions, count * sizeof(float3_custom)));
    CUDA_CHECK(cudaMalloc(&particles.velocities, count * sizeof(float3_custom)));
    CUDA_CHECK(cudaMalloc(&particles.radii, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.masses, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.restitutions, count * sizeof(float)));

    printf("Particle memory allocated: %.2f MB\n",
           count * (sizeof(float3_custom) * 2 + sizeof(float) * 3) / 1e6);

    initRandom(count);

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
        params.worldMax);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("✓ Initialized %d particles with diverse properties:\n", count);
    printf("  - Radius range: 0.05 - 0.40\n");
    printf("  - Mass range: varied by radius³ × density\n");
    printf("  - Velocity range: -8 to +8 (XY), 0 to +8 (Z)\n");
    printf("  - Restitution range: 0.3 - 0.95\n");
}

void PhysicsSimulator::integrate(ParticleSystem &particles, float dt)
{
    int numBlocks = (particles.count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    integrateKernel<<<numBlocks, BLOCK_SIZE>>>(
        particles.positions,
        particles.velocities,
        particles.masses,
        particles.count,
        params.gravity,
        dt,
        params.damping);
    CUDA_CHECK_LAST_ERROR();
}

void PhysicsSimulator::handleBoundaryCollisions(ParticleSystem &particles)
{
    int numBlocks = (particles.count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    boundaryCollisionKernel<<<numBlocks, BLOCK_SIZE>>>(
        particles.positions,
        particles.velocities,
        particles.radii,
        particles.restitutions,
        particles.count,
        params.worldMin,
        params.worldMax,
        params.groundRestitution);
    CUDA_CHECK_LAST_ERROR();
}