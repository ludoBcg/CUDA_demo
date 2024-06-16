/*********************************************************************************************************************
 *
 * texture.cu
 *
 * Example code for CUDA - OpenGL Interoperability
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


// Sources:
//
// [1] https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP
// [2] https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g80d12187ae7590807c7676697d9fe03d
// [3] https://forums.developer.nvidia.com/t/writing-gl-textures-in-cuda-example-code-using-cudagraphicsglregisterimage-rather-than-pbo/15850
// [4] https://randallr.wordpress.com/2011/12/
//


#include "texture.h"

namespace cudademo
{

 /*--------------------------------------------------------------------------------------------------------+
 |                                               HOST CODE                                                 |
 +--------------------------------------------------------------------------------------------------------*/


// Creates an OpenGL 3D texture
void createTex3D(GLuint& _tex3D)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glGenTextures(1, &_tex3D);
    glBindTexture(GL_TEXTURE_3D, _tex3D);
    {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
 
        // int 8 rgba texture
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8UI, 16, 16, 16, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL /*data*/);
    }

    glBindTexture(GL_TEXTURE_3D, 0);
}

void testTexture(const GLuint& _tex3D, cudaGraphicsResource* _texCUDA)
{
    cudaError_t err =  cudaGraphicsGLRegisterImage( &_texCUDA, 
                                                    _tex3D, 
                                                    GL_TEXTURE_3D,
                                                    cudaGraphicsMapFlagsNone);

    switch (err)
    {
    case cudaSuccess:
        std::cout << "cudaGraphicsGLRegisterImage: cudaSuccess" << std::endl;
        break;
    case cudaErrorInvalidDevice:
        std::cerr << "cudaGraphicsGLRegisterImage error: cudaErrorInvalidDevice" << std::endl;
        break;
    case cudaErrorInvalidValue:
        std::cerr << "cudaGraphicsGLRegisterImage error: cudaErrorInvalidValue" << std::endl;
        break;
    case cudaErrorInvalidResourceHandle:
        std::cerr << "cudaGraphicsGLRegisterImage error: cudaErrorInvalidResourceHandle" << std::endl;
        break;
    case cudaErrorUnknown:
        std::cerr << "cudaGraphicsGLRegisterImage error: cudaErrorUnknown" << std::endl;
        break;
    default:
        std::cerr << "cudaGraphicsGLRegisterImage error: " << err << std::endl;
        break;
    }

    return;
}

} // namespace cudademo