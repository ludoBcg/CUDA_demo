/*********************************************************************************************************************
 *
 * basics.cu
 *
 * CUDA example code for basic kernels
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/

// Sources:
//
// [1] https://developer.nvidia.com/blog/even-easier-introduction-cuda/
// [2] https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
// [3] https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
// [4] https://developer.nvidia.com/blog/six-ways-saxpy/
//


#include "basics.h"

namespace cudademo
{

 /*--------------------------------------------------------------------------------------------------------+
 |                                                 KERNELS                                                 |
 |                                         Device code (i.e., GPU)                                         |
 +--------------------------------------------------------------------------------------------------------*/

// __global__ keyword tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from host code


/* 
 * Kernel addV1 :
 * adds the elements of two arrays X and Y
 * writes the sum in Y
 * dummy version (cf. [1])
 */
__global__
void addV1(int _n, float *_x, float *_y)
{
    for (int i = 0; i < _n; i++)
        _y[i] = _x[i] + _y[i];
}

/* 
 * Kernel addV2 :
 * single-block, threaded version (cf. [1])
 */
__global__
void addV2(int _n, float *_x, float *_y)
{
    /*
    *    Example with 1 block of 256 threads:
    *
    *             Block 1
    *     __________/\_______
    *    /                   \
    *    | T0 | T1 |....|T255|
    *
    *  arrayX/Y:
    *    | 0  | 1  |....|255 | 256 |257.|....|
    *           |                    |
    *           V                    V
    *           i = tIdx (here T1)   i = tIdx+stride
    *            \__________________/
    *            stride = Block size
    */

    int index = threadIdx.x; // starting idx = current thread id
    int stride = blockDim.x; // stride = total nb of threads in this block

    for (int i = index; i < _n; i += stride)
        _y[i] = _x[i] + _y[i];
}

/* 
 * Kernel addV3 :
 * multi-block, threaded version (cf. [1])
 */
__global__
void addV3(int _n, float *_x, float *_y)
{
    /*
    *    Example with blocks of 256 threads:
    *
    *                            GRID
    *     ________________________________/\_______________________________
    *    /                                                                 \
    *             Block 1              Block 2      ....        Block 3907
    *     __________/\_______  __________/\_______        __________/\_______
    *    /                   \/                   \ .... /                   \
    *    | T0 | T1 |....|T255|| T0 |....|....|T255||....|| T0 |....|....|T255|
    *
    *  arrayX/Y:
    *    | 0  | 1  |....|....|.....|....|....|....|.....|.....|....|....|....|....
    *           |                                                              |
    *           V                                                              V
    *           i = tIdx (here T1)                                             i = tIdx+stride
    *            \____________________________________________________________/
    *                            stride = nb of thread in grid
    */

    // threadIdx.x                           is the index of current thread IN ITS BLOCK (i.e., always in [0;255])
    // blockIdx.x * blockDim.x               is the global index of first thread in current block
    // blockIdx.x * blockDim.x + threadIdx.x is the global index of current thread in the entire grid

    // blockDim.x                            is the number of threads in a block
    // gridDim.x                             is the number of blocks in the grid

    int index = blockIdx.x * blockDim.x + threadIdx.x; // index = global index of current thread in the entire grid
    int stride = blockDim.x * gridDim.x;               // stride = total nb of threads in the grid

    // this loop should be executed once since stride >= _n ...
    for (int i = index; i < _n; i += stride)           
        _y[i] = _x[i] + _y[i];

    // ...therefore could be replaced by:
    //if (index < _n) 
    //  _y[index] = _x[index] + _y[index];
}


//SAXPY stands for "Single-precision A*X Plus Y" (cf. [4])
__global__
void saxpy(int _n, float _a, float* _x, float* _y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < _n) 
      _y[index] = _a * _x[index] + _y[index];
}



 /*--------------------------------------------------------------------------------------------------------+
 |                                               HOST CODE                                                 |
 |                                              (i.e., CPU)                                                |
 +--------------------------------------------------------------------------------------------------------*/


void testBasics()
{
    GLtools::CpuTimer cpuChrono;
    GLtools::GpuTimer gpuChrono;

    int N = 1<<20; // N = 1M elements
    std::cout << "Number of elements: " << N << std::endl;


    {
        // GPU allocation of two arrays
        float *arrayX, *arrayY;
        cudaMallocManaged(&arrayX, N * sizeof(float));  //CPU equivalent: float *arrayX = new float[N];
        cudaMallocManaged(&arrayY, N * sizeof(float));  //CPU equivalent: float *arrayY = new float[N];
        // Using cudaMallocManaged(), arrays are automatically managed by Unified Memory system (cf. Note 1 below)

        // initialize arrayX and arrayY on the host
        for (int i = 0; i < N; i++)
        {
            arrayX[i] = 1.0f;
            arrayY[i] = 2.0f;
        }

        // Run kernels on 1M elements on the CPU

        // use device timer instead of host timer (cf. Note 2 below)
        gpuChrono.start("kernel add1");

        // Kernel 1: add() single-threaded
        addV1 <<<1, 1>>> (N, arrayX, arrayY);

        gpuChrono.stop();
        gpuChrono.start("kernel add2");

        // Kernel 2: add() using 1 block of 256 threads (size of blocks must be a multiple of 32)
        addV2 <<<1, 256>>> (N, arrayX, arrayY);

        gpuChrono.stop();
        gpuChrono.start("kernel add3");

        // Kernel 3: add() using as many blocks of 256 threads as necessary
        int blockSize = 256;
        // number of block necessary is number of element to process / blockSize
        // (rounded up in case N is not a multiple of blockSize)
        int numBlocks = (N + blockSize - 1) / blockSize;
        addV3 <<<numBlocks, blockSize>>> (N, arrayX, arrayY);

        gpuChrono.stop();
        gpuChrono.start("kernel saxpy");

        // Kernel 4: saxpy() using as many blocks of 256 threads as necessary
        saxpy <<<numBlocks, blockSize>>> (N, 2.0f, arrayX, arrayY);

        gpuChrono.stop();

        // tells CPU to wait for GPU threads
        cudaDeviceSynchronize();

        // Check for errors (all values should be 7.0f)
        float maxError = 0.0f;
        for (int i = 0; i < N; i++)
            maxError = fmax(maxError, fabs(arrayY[i] - 7.0f));
        std::cout << std::endl << "Max error: " << maxError << std::endl;


        // GPU Free memory
        cudaFree(arrayX);  //CPU equivalent: delete [] arrayX;
        cudaFree(arrayY);  //CPU equivalent: delete [] arrayY;
    }


    // -----------------------------------------------------------------------------------------------------
    // Note 1:
    //
    // Usage of cudaMallocManaged() automatically makes arrayX and arrayY available on device and host,
    // and spares us the Host-To-Device and Device-To-Host copies as below 
    // (cf. [1] for cudaMallocManaged() version, and [2] for cudaMalloc() + cudaMemcpy() version):
    /*
    {
        float *h_arrayX, *h_arrayY; // CPU / host arrays
        // host allocation
        h_arrayX = (float*)malloc(N * sizeof(float));
        h_arrayY = (float*)malloc(N * sizeof(float));
        // fills-in values
        for (int i = 0; i < N; i++) 
        {
            h_arrayX[i] = 1.0f;
            h_arrayY[i] = 2.0f;
        }

        float *d_arrayX, *d_arrayY; // GPU / device arrays
        // device allocation
        cudaMalloc(&d_arrayX, N * sizeof(float));
        cudaMalloc(&d_arrayY, N * sizeof(float));
        // copies values from host to device
        cudaMemcpy(d_arrayX, h_arrayX, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_arrayY, h_arrayY, N * sizeof(float), cudaMemcpyHostToDevice);

        // kernel execution
        saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_arrayX, d_arrayY);

        // copies back result values from device to host
        cudaMemcpy(h_arrayY, d_arrayY, N * sizeof(float), cudaMemcpyDeviceToHost);

        float maxError = 0.0f;
        for (int i = 0; i < N; i++)
            maxError = fmax(maxError, fabs(h_arrayY[i] - 4.0f));
        std::cout << "Max error: " << maxError << std::endl;

        // free device memory
        cudaFree(d_arrayX);
        cudaFree(d_arrayY);
        // free host memory
        free(h_arrayX);
        free(h_arrayY);
    }
    */


    // -----------------------------------------------------------------------------------------------------
    // Note 2:
    //
    // Usage of a cpuTimer requires explicit synchronization barrier cudaDeviceSynchronize() to block CPU 
    // execution until all previously issued commands on the device have completed, which stalls the GPU pipeline.
    // Without this barrier, this code would measure the kernel launch time and not the kernel execution time.
    // Therefore, the code above uses a gpuTimer based on CUDA Events (cf. [3] for details).
    // The cpuTimer version would look like the following:
    /*
    {
        // GPU allocation of two arrays
        float *arrayX, *arrayY;
        cudaMallocManaged(&arrayX, N * sizeof(float));
        cudaMallocManaged(&arrayY, N * sizeof(float));
        
        for (int i = 0; i < N; i++) {
            arrayX[i] = 1.0f;
            arrayY[i] = 2.0f;
        }

        cpuChrono.start("kernel add1");

        // Kernel 1: add() single-threaded
        addV1 << <1, 1 >> > (N, arrayX, arrayY);

        // tells CPU to wait for GPU threads
        cudaDeviceSynchronize();
        cpuChrono.stop();
        cpuChrono.start("kernel add2");

        addV2 << <1, 256 >> > (N, arrayX, arrayY);

        cudaDeviceSynchronize();
        cpuChrono.stop();

        float maxError = 0.0f;
        for (int i = 0; i < N; i++)
            maxError = fmax(maxError, fabs(arrayY[i] - 4.0f));
        std::cout << std::endl << "Max error: " << maxError << std::endl;

        cudaFree(arrayX);
        cudaFree(arrayY);
    }
    */

    return;
}

} // namespace cudademo