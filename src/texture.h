/*********************************************************************************************************************
 *
 * texture.h
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#ifndef TEXTURE_H_ 
#define TEXTURE_H_ 

#include "utils.h"

namespace cudademo
{
    void createTex3D(GLuint& _tex3D);
    void testTexture(const GLuint& _tex3D, cudaGraphicsResource* _texCUDA);

} // namespace cudademo

#endif // TEXTURE_H_ 
