#include "physics.cuh"
#include <curand_kernel.h>
#include <stdio.h>

// ==================== CUDA核函数 ====================

// 初始化随机数生成器
__global__ void initRandKernel(curandState *states, unsigned long seed, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    curand_init(seed, idx, 0, &states[idx]);
}

// 随机初始化粒子
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
    masses[idx] = 4.0f / 3.0f * 3.14159f * r * r * r * 1000.0f; // 密度1000

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

    // Y边界（地面使用特殊弹性系数）
    if (pos.y - radius < worldMin.y)
    {
        pos.y = worldMin.y + radius;
        vel.y = -vel.y * groundRestitution;
        vel.x *= 0.98f; // 地面摩擦
        vel.z *= 0.98f;
    }
    if (pos.y + radius > worldMax.y)
    {
        pos.y = worldMax.y - radius;
        vel.y = -vel.y * restitution;
    }

    // Z边界
    if (pos.z - radius < worldMin.z)
    {
        pos.z = worldMin.z + radius;
        vel.z = -vel.z * restitution;
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
    cudaMalloc(&d_randStates, count * sizeof(curandState));

    int numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initRandKernel<<<numBlocks, BLOCK_SIZE>>>(d_randStates, time(nullptr), count);
    cudaDeviceSynchronize();
}

void PhysicsSimulator::initializeParticles(ParticleSystem &particles, int count)
{
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
        params.worldMax);

    cudaDeviceSynchronize();
    printf("Initialized %d particles\n", count);
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
}