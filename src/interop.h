/*********************************************************************************************************************
 *
 * interop.h
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#ifndef INTEROP_H_ 
#define INTEROP_H_ 

#include "GLtools.h"

#include <cuda_gl_interop.h>


namespace cudademo
{
    void printErrors(cudaError_t _cudaErr = cudaSuccess);

    void mapPBO(GLuint& _volPBO, GLuint& _volTex, unsigned int _imByteSize,
                unsigned int _imWidth, unsigned int _imHeight, unsigned int _imDepth);

} // namespace cudademo

#endif // INTEROP_H_ 