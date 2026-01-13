#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#include "collision_detection.cuh"
#include <curand_kernel.h>

// 物理仿真参数
struct SimulationParams
{
    float3_custom gravity;  // 重力加速度
    float3_custom worldMin; // 世界边界
    float3_custom worldMax;
    float damping;           // 阻尼系数
    float groundRestitution; // 地面弹性系数
};

// 物理仿真器类
class PhysicsSimulator
{
public:
    PhysicsSimulator(const SimulationParams &params);
    ~PhysicsSimulator();

    // 初始化粒子系统
    void initializeParticles(ParticleSystem &particles, int count);

    // 积分更新位置和速度
    void integrate(ParticleSystem &particles, float dt);

    // 处理边界碰撞
    void handleBoundaryCollisions(ParticleSystem &particles);

    // 获取仿真参数
    const SimulationParams &getParams() const { return params; }

private:
    SimulationParams params;
    curandState *d_randStates;

    void initRandom(int count);
};

#endif // PHYSICS_CUH