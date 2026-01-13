#include "collision_detection.cuh"
#include "physics.cuh"
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

// 导出数据到文件
void exportFrame(const ParticleSystem &particles, int frameNum, const std::string &outputDir)
{
    // 从GPU复制数据到CPU
    std::vector<float3_custom> positions(particles.count);
    std::vector<float> radii(particles.count);

    cudaMemcpy(positions.data(), particles.positions,
               particles.count * sizeof(float3_custom), cudaMemcpyDeviceToHost);
    cudaMemcpy(radii.data(), particles.radii,
               particles.count * sizeof(float), cudaMemcpyDeviceToHost);

    // 写入文件
    char filename[256];
    sprintf(filename, "%s/frame_%04d.txt", outputDir.c_str(), frameNum);

    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        printf("Failed to open file: %s\n", filename);
        return;
    }

    fprintf(fp, "%d\n", particles.count);
    for (int i = 0; i < particles.count; i++)
    {
        fprintf(fp, "%.6f %.6f %.6f %.6f\n",
                positions[i].x, positions[i].y, positions[i].z, radii[i]);
    }

    fclose(fp);
}

// 性能测试
void performanceTest()
{
    printf("\n========== Performance Test ==========\n");

    int testCounts[] = {1000, 5000, 10000, 20000, 50000};

    SimulationParams params;
    params.gravity = float3_custom(0, -9.8f, 0);
    params.worldMin = float3_custom(-10, 0, -10);
    params.worldMax = float3_custom(10, 20, 10);
    params.damping = 0.999f;
    params.groundRestitution = 0.6f;

    for (int count : testCounts)
    {
        printf("\nTesting with %d particles...\n", count);

        // 创建仿真器和碰撞检测器
        PhysicsSimulator simulator(params);
        CollisionDetector detector(count * 2, params.worldMin, params.worldMax, 0.6f);

        // 初始化粒子
        ParticleSystem particles;
        simulator.initializeParticles(particles, count);

        // 预热
        for (int i = 0; i < 10; i++)
        {
            detector.updateGrid(particles);
            detector.detectAndResolveCollisions(particles, 0.016f);
        }

        // 测试100帧
        auto startTime = std::chrono::high_resolution_clock::now();

        float totalGridTime = 0;
        float totalCollisionTime = 0;
        int totalCollisions = 0;

        for (int frame = 0; frame < 100; frame++)
        {
            simulator.integrate(particles, 0.016f);
            simulator.handleBoundaryCollisions(particles);

            detector.updateGrid(particles);
            detector.detectAndResolveCollisions(particles, 0.016f);

            auto stats = detector.getStats();
            totalGridTime += stats.gridUpdateTime;
            totalCollisionTime += stats.collisionDetectionTime;
            totalCollisions += stats.collisionCount;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();

        printf("  Total time: %.2f ms (%.2f FPS)\n", totalTime, 100000.0f / totalTime);
        printf("  Avg grid update: %.3f ms\n", totalGridTime / 100);
        printf("  Avg collision detection: %.3f ms\n", totalCollisionTime / 100);
        printf("  Avg collisions per frame: %d\n", totalCollisions / 100);

        // 清理
        cudaFree(particles.positions);
        cudaFree(particles.velocities);
        cudaFree(particles.radii);
        cudaFree(particles.masses);
        cudaFree(particles.restitutions);
    }
}

int main(int argc, char **argv)
{
    printf("GPU Collision Detection System\n");
    printf("==============================\n\n");

    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        printf("No CUDA devices found!\n");
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n\n", prop.totalGlobalMem / 1e9);

    // 运行性能测试
    if (argc > 1 && std::string(argv[1]) == "--test")
    {
        performanceTest();
        return 0;
    }

    // 仿真参数
    SimulationParams params;
    params.gravity = float3_custom(0, -9.8f, 0);
    params.worldMin = float3_custom(-10, 0, -10);
    params.worldMax = float3_custom(10, 20, 10);
    params.damping = 0.999f;
    params.groundRestitution = 0.6f;

    // 创建仿真器和碰撞检测器
    int numParticles = 5000;
    PhysicsSimulator simulator(params);
    CollisionDetector detector(numParticles * 2, params.worldMin, params.worldMax, 0.6f);

    // 初始化粒子系统
    ParticleSystem particles;
    simulator.initializeParticles(particles, numParticles);

    // 创建输出目录
    system("mkdir output 2>nul");

    // 仿真循环
    float dt = 1.0f / 60.0f; // 60 FPS
    int totalFrames = 600;   // 10秒动画
    int exportInterval = 2;  // 每2帧导出一次

    printf("Starting simulation...\n");
    printf("Simulating %d frames at %.1f FPS\n", totalFrames, 1.0f / dt);
    printf("Output directory: ./output/\n\n");

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < totalFrames; frame++)
    {
        // 物理积分
        simulator.integrate(particles, dt);

        // 边界碰撞
        simulator.handleBoundaryCollisions(particles);

        // 更新空间网格
        detector.updateGrid(particles);

        // 碰撞检测和响应
        detector.detectAndResolveCollisions(particles, dt);

        // 导出帧数据
        if (frame % exportInterval == 0)
        {
            exportFrame(particles, frame / exportInterval, "output");
        }

        // 显示进度
        if (frame % 60 == 0)
        {
            auto stats = detector.getStats();
            printf("Frame %d/%d - Grid: %.2fms, Collision: %.2fms, Collisions: %d\n",
                   frame, totalFrames, stats.gridUpdateTime,
                   stats.collisionDetectionTime, stats.collisionCount);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    float totalTime = std::chrono::duration<float>(endTime - startTime).count();

    printf("\nSimulation complete!\n");
    printf("Total time: %.2f seconds (%.2f FPS)\n", totalTime, totalFrames / totalTime);
    printf("Frames exported: %d\n", totalFrames / exportInterval);

    // 清理
    cudaFree(particles.positions);
    cudaFree(particles.velocities);
    cudaFree(particles.radii);
    cudaFree(particles.masses);
    cudaFree(particles.restitutions);

    return 0;
}