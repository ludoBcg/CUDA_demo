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
    void printErrors(cudaError_t _cudaErr = cudaSuccess);


    void createTex3D(GLuint& _tex3D);
    void testTexture(const GLuint& _tex3D, cudaGraphicsResource* _texCUDA);


    void createTex2D(GLuint& _tex2D, const int _imWidth, const int _imHeight,
                     const std::vector<unsigned char>& _imgData );
    void createPBO(GLuint& _tex2D, const int _imTotalDim,
                   const std::vector<unsigned char>& _imgData );

    void testTextureObject(const std::string& _filenameIn, const std::string& _filenameOut);
    void testPBO(const std::string& _filenameIn, const std::string& _filenameOut);

} // namespace cudademo

#endif // TEXTURE_H_ 
