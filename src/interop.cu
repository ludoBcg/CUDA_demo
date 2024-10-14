/*********************************************************************************************************************
 *
 * interop.cu
 *
 * Example code for CUDA - OpenGL Interoperability
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#include "interop.h"

namespace cudademo
{

    struct cudaGraphicsResource* m_graphResCUDA;


    __global__
    void simpleKernelPBO3D(int _dimX, int _dimY, int _dimZ,unsigned char* _inout)
    {

        int idX = blockIdx.x * blockDim.x + threadIdx.x;
        int idY = blockIdx.y * blockDim.y + threadIdx.y;
        int idZ = blockIdx.z * blockDim.z + threadIdx.z;
        int strideX = blockDim.x * gridDim.x;
        int strideY = blockDim.y * gridDim.y;
        int strideZ = blockDim.z * gridDim.z;

        for (int i = idX; i < _dimX; i += strideX)
        {
            for (int j = idY; j < _dimY; j += strideY)
            {
                for (int k = idZ; k < _dimZ; k += strideZ)
                {
                    int index =   (k * _dimY * _dimX) 
                                + (j * _dimX) 
                                + i;

                    // overwrite value
                    _inout[index] = (255 - _inout[index]);
                }
            }
        }
    }


	void mapPBO(GLuint& _volPBO, GLuint& _volTex, unsigned int _imByteSize,
                unsigned int _imWidth, unsigned int _imHeight, unsigned int _imDepth)
	{
		// 1. Register the PBO as a CUDA graphics resource -----------------------------
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&m_graphResCUDA, _volPBO, cudaGraphicsMapFlagsNone); 
        printErrors(err);

        // 2. map the CUDA graphic resource to a pointer -------------------------------
        unsigned char* cuda_mapped_ptr = nullptr;
        size_t numBytes = _imByteSize;
        cudaMalloc((void**)&cuda_mapped_ptr, numBytes);
        err = cudaGraphicsMapResources(1, &m_graphResCUDA, 0);
        printErrors(err);
        size_t size;
        err = cudaGraphicsResourceGetMappedPointer( (void **)&cuda_mapped_ptr, &size, m_graphResCUDA);
        printErrors(err);


        // 4. Execute kernel, using CUDA mapped ptr as input ---------------------------
        dim3 blockSize(8, 8, 8);
        dim3 numBlocks((_imWidth + blockSize.x - 1) / blockSize.x, 
                       (_imHeight + blockSize.y - 1) / blockSize.y,
                       (_imDepth + blockSize.z - 1) / blockSize.z );
        simpleKernelPBO3D<<<numBlocks, blockSize>>> (_imWidth, _imHeight, _imDepth, cuda_mapped_ptr);
        cudaDeviceSynchronize();

        // 6. Finally unmap resource and free memory -----------------------------------
        err = cudaGraphicsUnmapResources(1, &m_graphResCUDA, 0);
        printErrors(err);


        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _volPBO);
        glBindTexture(GL_TEXTURE_3D, _volTex);
        glTexSubImage3D(GL_TEXTURE_3D, // target
                        0, // level
                        0, // x offset
                        0, // y offset
                        0, // z offset
                        _imWidth,
                        _imHeight,
                        _imDepth,
                        GL_RED, // format
                        GL_UNSIGNED_BYTE, // type
                        nullptr); // zeroed memory
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        
        cudaFree(cuda_mapped_ptr);

        printErrors();
	}


    /* 
    * Prints CUDA / OpenGL errors if not successful
    */
    void printErrors(cudaError_t _cudaErr)
    {
	    // prints last OpenGL error
	    if (glGetError() != 0) { std::cerr << "OpenGL error : " << glGetError() << std::endl; }

	    // prints CUDA error
	    if (_cudaErr != cudaSuccess) {std::cerr << "CUDA error: " << cudaGetErrorString(_cudaErr) << std::endl;}
    }
}