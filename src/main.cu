#include "collision_detection.cuh"
#include "physics.cuh"
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <direct.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>

// ==================== User Interaction Functions ====================

void printWelcomeBanner()
{
    printf("\n");
    printf("================================================================\n");
    printf("                                                                \n");
    printf("      GPU Collision Detection System - Animation Project       \n");
    printf("         Fast Collision Detection for Animation                \n");
    printf("                                                                \n");
    printf("================================================================\n");
    printf("\n");
}

void printMainMenu()
{
    printf("\n");
    printf("================================================================\n");
    printf("                    Select Run Mode                            \n");
    printf("================================================================\n");
    printf("  1. Quick Demo (Recommended)  - 1000 particles, 600 frames    \n");
    printf("                                 (~10 seconds)                 \n");
    printf("  2. Standard Simulation       - 2000 particles, 600 frames    \n");
    printf("                                 (~20 seconds)                 \n");
    printf("  3. Large Scale Simulation    - 5000 particles, 1200 frames   \n");
    printf("                                 (~1 minute)                   \n");
    printf("  4. Performance Benchmark     - Test multiple particle counts \n");
    printf("  5. Custom Parameters         - Set all parameters manually   \n");
    printf("  0. Exit                                                       \n");
    printf("================================================================\n");
    printf("\nEnter your choice (0-5): ");
}

struct SimConfig
{
    int numParticles;
    int totalFrames;
    int targetFPS;
    int exportInterval;
    float gravityZ;
    float damping;
    float groundRestitution;
    std::string outputDir;
    std::string videoFilename;
    bool autoVideo;
    std::string description;
};

SimConfig getPresetConfig(int choice)
{
    SimConfig config;
    config.targetFPS = 60;
    config.gravityZ = -9.8f;
    config.damping = 0.999f;
    config.groundRestitution = 0.6f;
    config.outputDir = "output";
    config.autoVideo = true;

    switch (choice)
    {
    case 1: // Quick Demo
        config.numParticles = 1000;
        config.totalFrames = 600;
        config.exportInterval = 2;
        config.videoFilename = "demo_1000p.mp4";
        config.description = "Quick Demo - Fast preview";
        break;

    case 2: // Standard Simulation
        config.numParticles = 2000;
        config.totalFrames = 600;
        config.exportInterval = 2;
        config.videoFilename = "standard_2000p.mp4";
        config.description = "Standard Mode - Balanced quality and speed";
        break;

    case 3: // Large Scale
        config.numParticles = 5000;
        config.totalFrames = 1200;
        config.exportInterval = 3;
        config.videoFilename = "large_5000p.mp4";
        config.description = "Large Scale - Showcase GPU performance";
        break;

    default:
        config.numParticles = 1000;
        config.totalFrames = 600;
        config.exportInterval = 2;
        config.videoFilename = "animation.mp4";
        config.description = "Default Config";
        break;
    }

    return config;
}

SimConfig getCustomConfig()
{
    SimConfig config;

    printf("\n================================================================\n");
    printf("                   Custom Parameter Setup                      \n");
    printf("================================================================\n\n");

    printf("Number of particles (recommend: 1000-10000) [default: 2000]: ");
    std::string input;
    std::getline(std::cin, input);
    config.numParticles = input.empty() ? 2000 : std::stoi(input);

    printf("Total frames (recommend: 300-1800) [default: 600]: ");
    std::getline(std::cin, input);
    config.totalFrames = input.empty() ? 600 : std::stoi(input);

    printf("Target FPS (recommend: 30/60) [default: 60]: ");
    std::getline(std::cin, input);
    config.targetFPS = input.empty() ? 60 : std::stoi(input);

    printf("Export interval (export every N frames) [default: 2]: ");
    std::getline(std::cin, input);
    config.exportInterval = input.empty() ? 2 : std::stoi(input);

    printf("Gravity Z component (m/s^2) [default: -9.8]: ");
    std::getline(std::cin, input);
    config.gravityZ = input.empty() ? -9.8f : std::stof(input);

    printf("Damping coefficient (0-1) [default: 0.999]: ");
    std::getline(std::cin, input);
    config.damping = input.empty() ? 0.999f : std::stof(input);

    printf("Ground restitution (0-1) [default: 0.6]: ");
    std::getline(std::cin, input);
    config.groundRestitution = input.empty() ? 0.6f : std::stof(input);

    printf("Output directory [default: output]: ");
    std::getline(std::cin, input);
    config.outputDir = input.empty() ? "output" : input;

    printf("Video filename [default: custom.mp4]: ");
    std::getline(std::cin, input);
    config.videoFilename = input.empty() ? "custom.mp4" : input;

    printf("Auto generate video? (y/n) [default: y]: ");
    std::getline(std::cin, input);
    config.autoVideo = (input.empty() || input[0] == 'y' || input[0] == 'Y');

    config.description = "Custom Configuration";

    return config;
}

void printConfigSummary(const SimConfig &config)
{
    printf("\n");
    printf("================================================================\n");
    printf("                    Configuration Summary                      \n");
    printf("================================================================\n");
    printf(" Mode:              %s\n", config.description.c_str());
    printf(" Particles:         %d\n", config.numParticles);
    printf(" Total frames:      %d\n", config.totalFrames);
    printf(" Target FPS:        %d (time step: %.4fs)\n", config.targetFPS, 1.0f / config.targetFPS);
    printf(" Export interval:   every %d frames\n", config.exportInterval);
    printf(" Gravity:           %.2f m/s^2 (Z-axis)\n", config.gravityZ);
    printf(" Damping:           %.4f\n", config.damping);
    printf(" Ground elasticity: %.2f\n", config.groundRestitution);
    printf(" Output directory:  %s\n", config.outputDir.c_str());
    printf(" Video filename:    %s\n", config.videoFilename.c_str());
    printf(" Auto video:        %s\n", config.autoVideo ? "Yes" : "No");
    printf("================================================================\n");

    float estimatedTime = config.numParticles * config.totalFrames / 100000.0f;
    printf("\nEstimated simulation time: %.1f seconds\n", estimatedTime);
    printf("Estimated video duration:  %.1f seconds\n", config.totalFrames / (float)config.targetFPS);
    printf("\nPress Enter to continue...");
    std::cin.get();
}

// ==================== Original Functions (Unchanged) ====================

void exportFrame(const ParticleSystem &particles, int frameNum, const std::string &outputDir)
{
    std::vector<float3_custom> positions(particles.count);
    std::vector<float> radii(particles.count);

    CUDA_CHECK(cudaMemcpy(positions.data(), particles.positions,
                          particles.count * sizeof(float3_custom), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(radii.data(), particles.radii,
                          particles.count * sizeof(float), cudaMemcpyDeviceToHost));

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

void exportParticleStats(const ParticleSystem &particles, const std::string &filename)
{
    std::vector<float> radii(particles.count);
    std::vector<float> masses(particles.count);
    std::vector<float> restitutions(particles.count);
    std::vector<float3_custom> velocities(particles.count);

    CUDA_CHECK(cudaMemcpy(radii.data(), particles.radii,
                          particles.count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(masses.data(), particles.masses,
                          particles.count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(restitutions.data(), particles.restitutions,
                          particles.count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velocities.data(), particles.velocities,
                          particles.count * sizeof(float3_custom), cudaMemcpyDeviceToHost));

    FILE *fp = fopen(filename.c_str(), "w");
    if (!fp)
        return;

    fprintf(fp, "Particle_ID,Radius,Mass,Restitution,Velocity_X,Velocity_Y,Velocity_Z,Speed\n");
    for (int i = 0; i < particles.count; i++)
    {
        float speed = velocities[i].length();
        fprintf(fp, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                i, radii[i], masses[i], restitutions[i],
                velocities[i].x, velocities[i].y, velocities[i].z, speed);
    }

    fclose(fp);
    printf(">> Particle statistics exported to %s\n", filename.c_str());
}

void performanceTest()
{
    printf("\n================================================================\n");
    printf("                   Performance Benchmark                       \n");
    printf("================================================================\n\n");

    int testCounts[] = {1000, 5000, 10000, 20000};

    SimulationParams params;
    params.gravity = float3_custom(0, 0, -9.8f);
    params.worldMin = float3_custom(-10, -10, 0);
    params.worldMax = float3_custom(10, 10, 20);
    params.damping = 0.999f;
    params.groundRestitution = 0.6f;

    for (int count : testCounts)
    {
        printf("\n========================================\n");
        printf("Testing with %d particles...\n", count);
        printf("========================================\n");

        PhysicsSimulator simulator(params);
        CollisionDetector detector(count * 2, params.worldMin, params.worldMax, 0.6f);

        ParticleSystem particles;
        simulator.initializeParticles(particles, count);

        printf("Warming up...\n");
        for (int i = 0; i < 10; i++)
        {
            detector.updateGrid(particles);
            detector.detectAndResolveCollisions(particles, 0.016f);
        }

        printf("Running benchmark (100 frames)...\n");
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

        printf("\n--- Results ---\n");
        printf("  Total time:           %.2f ms (%.2f FPS)\n", totalTime, 100000.0f / totalTime);
        printf("  Avg grid update:      %.3f ms\n", totalGridTime / 100);
        printf("  Avg collision detect: %.3f ms\n", totalCollisionTime / 100);
        printf("  Avg collisions/frame: %d\n", totalCollisions / 100);

        cudaFree(particles.positions);
        cudaFree(particles.velocities);
        cudaFree(particles.radii);
        cudaFree(particles.masses);
        cudaFree(particles.restitutions);
    }

    printf("\n========== Performance Test Complete ==========\n\n");
}

void generateVideo(const std::string &outputDir, const std::string &videoFilename)
{
    printf("\n================================================\n");
    printf("  Generating MP4 Video...\n");
    printf("================================================\n");

    int result = system("..\\venv\\Scripts\\python.exe --version >nul 2>&1");
    if (result != 0)
    {
        printf("Warning: venv Python not found! Skipping video generation.\n");
        return;
    }

    char cmd[512];
    sprintf(cmd, "..\\venv\\Scripts\\python.exe visualize.py %s %s",
            outputDir.c_str(), videoFilename.c_str());

    printf(">> Calling Python visualization script...\n");
    printf("   Command: %s\n", cmd);

    result = system(cmd);

    FILE *checkFile = fopen(videoFilename.c_str(), "rb");
    bool fileExists = (checkFile != nullptr);
    if (checkFile)
        fclose(checkFile);

    if (fileExists)
    {
        printf("\n>> Video generation complete!\n");
        printf("   Output: %s\n", videoFilename.c_str());
    }
    else
    {
        printf("\n!! Video generation failed!\n");
        printf("   Exit code: %d\n", result);
        printf("   Output file not found: %s\n", videoFilename.c_str());
    }
}

void runSimulation(const SimConfig &config)
{
    float dt = 1.0f / config.targetFPS;

    SimulationParams params;
    params.gravity = float3_custom(0, 0, config.gravityZ);
    params.worldMin = float3_custom(-10, -10, 0);
    params.worldMax = float3_custom(10, 10, 20);
    params.damping = config.damping;
    params.groundRestitution = config.groundRestitution;

    PhysicsSimulator simulator(params);
    CollisionDetector detector(config.numParticles * 2, params.worldMin, params.worldMax, 0.6f);

    ParticleSystem particles;
    simulator.initializeParticles(particles, config.numParticles);

    _mkdir(config.outputDir.c_str());

    exportParticleStats(particles, config.outputDir + "/particle_properties.csv");

    printf("\n================================================\n");
    printf("  Simulation Running...\n");
    printf("================================================\n\n");

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < config.totalFrames; frame++)
    {
        simulator.integrate(particles, dt);
        CUDA_CHECK_LAST_ERROR();

        simulator.handleBoundaryCollisions(particles);
        CUDA_CHECK_LAST_ERROR();

        for (int iter = 0; iter < 5; iter++)
        {
            detector.updateGrid(particles);
            CUDA_CHECK_LAST_ERROR();

            detector.detectAndResolveCollisions(particles, dt);
            CUDA_CHECK_LAST_ERROR();
        }

        if (frame % config.exportInterval == 0)
        {
            exportFrame(particles, frame / config.exportInterval, config.outputDir);
        }

        if (frame % 60 == 0)
        {
            auto stats = detector.getStats();
            float progress = (frame + 1) * 100.0f / config.totalFrames;
            printf("[Progress %5.1f%%] Frame %4d/%4d | Grid: %5.2fms | Collision: %5.2fms | Count: %5d\n",
                   progress, frame, config.totalFrames, stats.gridUpdateTime,
                   stats.collisionDetectionTime, stats.collisionCount);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    float totalTime = std::chrono::duration<float>(endTime - startTime).count();

    printf("\n================================================\n");
    printf("  Simulation Complete!\n");
    printf("================================================\n");
    printf("Total time:      %.2f seconds\n", totalTime);
    printf("Average FPS:     %.2f\n", config.totalFrames / totalTime);
    printf("Frames exported: %d\n", config.totalFrames / config.exportInterval);
    printf("================================================\n");

    if (config.autoVideo)
    {
        generateVideo(config.outputDir, config.videoFilename);
    }

    cudaFree(particles.positions);
    cudaFree(particles.velocities);
    cudaFree(particles.radii);
    cudaFree(particles.masses);
    cudaFree(particles.restitutions);

    printf("\n>> All tasks complete!\n");
}

// ==================== Main Function ====================

int main(int argc, char **argv)
{
    printWelcomeBanner();

    // Check GPU device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("ERROR: No CUDA devices found!\n");
        return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n\n", prop.totalGlobalMem / 1e9);

    // Command line mode (backward compatible)
    if (argc > 1)
    {
        if (std::string(argv[1]) == "--test")
        {
            performanceTest();
            return 0;
        }

        // Full command line arguments
        if (argc >= 11)
        {
            SimConfig config;
            config.numParticles = atoi(argv[1]);
            config.totalFrames = atoi(argv[2]);
            config.targetFPS = atoi(argv[3]);
            config.exportInterval = atoi(argv[4]);
            config.gravityZ = (float)atof(argv[5]);
            config.damping = (float)atof(argv[6]);
            config.groundRestitution = (float)atof(argv[7]);
            config.outputDir = argv[8];
            config.videoFilename = argv[9];
            config.autoVideo = (atoi(argv[10]) == 1);
            config.description = "Command-line mode";

            printConfigSummary(config);
            runSimulation(config);
            return 0;
        }
    }

    // Interactive menu mode
    while (true)
    {
        printMainMenu();

        int choice;
        std::cin >> choice;
        std::cin.ignore(); // Clear input buffer

        if (choice == 0)
        {
            printf("\nThank you for using! Goodbye!\n");
            break;
        }

        SimConfig config;

        if (choice == 4)
        {
            performanceTest();
            printf("\nPress Enter to return to main menu...");
            std::cin.get();
            continue;
        }
        else if (choice == 5)
        {
            config = getCustomConfig();
        }
        else if (choice >= 1 && choice <= 3)
        {
            config = getPresetConfig(choice);
        }
        else
        {
            printf("\nInvalid option! Please try again.\n");
            continue;
        }

        printConfigSummary(config);
        runSimulation(config);

        printf("\nPress Enter to return to main menu...");
        std::cin.get();
    }

    return 0;
}