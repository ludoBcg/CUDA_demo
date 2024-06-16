/*********************************************************************************************************************
 *
 * utils.h
 *
 * Timers structures
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#ifndef UTILS_H_ 
#define UTILS_H_ 

#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>

#include <lodepng.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace cudademo
{
	// Host timer using on std::chrono
    struct cpuTimer
    {
        std::chrono::time_point<std::chrono::system_clock> tStart;

        void start(const  std::string& msg)
        {
            std::cout << std::endl << "[Host timer] Starting " << msg << " ..." << std::endl;
            tStart = std::chrono::system_clock::now();
        }

        void stop()
        {
            auto tStop = std::chrono::system_clock::now();
            std::cout << "[Host timer] ... finished in " 
                      << std::chrono::duration_cast<std::chrono::milliseconds>(tStop - tStart).count() 
                      << " ms" << std::endl;
        }
    };


    // Device timer using cudaEvent_t
    struct gpuTimer
    {
        // uses CUDA events to measure time without cudaDeviceSynchronize() (cf. [3])
        cudaEvent_t tStart, tStop;

        gpuTimer()
        {
            cudaEventCreate(&tStart);
            cudaEventCreate(&tStop);
        }

        void start(const  std::string& msg)
        {
            std::cout << std::endl << "[Device timer] Starting " << msg << " ..." << std::endl;
            cudaEventRecord(tStart);
        }

        void stop()
        {
            cudaEventRecord(tStop);
            cudaEventSynchronize(tStop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, tStart, tStop);
            std::cout << "[Device timer] ... finished in " 
                      << milliseconds 
                      << " ms" << std::endl;
        }
    };


} // namespace cudademo

#endif // UTILS_H_ 