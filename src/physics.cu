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

// 随机初始化粒子（修改：Z轴为高度）
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

    // 随机位置（Z轴为高度）
    float x = worldMin.x + curand_uniform(&localState) * (worldMax.x - worldMin.x);
    float y = worldMin.y + curand_uniform(&localState) * (worldMax.y - worldMin.y);
    float z = worldMin.z + 0.3f * (worldMax.z - worldMin.z) +
              curand_uniform(&localState) * 0.6f * (worldMax.z - worldMin.z);
    positions[idx] = float3_custom(x, y, z);

    // 随机速度
    float vx = (curand_uniform(&localState) - 0.5f) * 5.0f;
    float vy = (curand_uniform(&localState) - 0.5f) * 5.0f;
    float vz = curand_uniform(&localState) * 2.0f; // Z方向向上的初速度
    velocities[idx] = float3_custom(vx, vy, vz);

    // 随机半径 (0.1 到 0.3)
    radii[idx] = 0.1f + curand_uniform(&localState) * 0.2f;

    // 质量与半径立方成正比
    float r = radii[idx];
    masses[idx] = 4.0f / 3.0f * 3.14159f * r * r * r * 1000.0f;

    // 随机弹性系数 (0.5 到 0.95)
    restitutions[idx] = 0.5f + curand_uniform(&localState) * 0.45f;

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

// 处理边界碰撞（修改：Z轴为高度）
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

    // Z边界（修改：Z为高度，地面在Z=worldMin.z）
    if (pos.z - radius < worldMin.z)
    {
        pos.z = worldMin.z + radius;
        vel.z = -vel.z * groundRestitution;
        vel.x *= 0.98f; // 地面摩擦
        vel.y *= 0.98f;
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

    printf("Initialized %d particles (Z-axis UP coordinate system)\n", count);
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