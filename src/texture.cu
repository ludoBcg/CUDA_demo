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
// [1] https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html
// [2] https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html
// [3] https://forums.developer.nvidia.com/t/writing-gl-textures-in-cuda-example-code-using-cudagraphicsglregisterimage-rather-than-pbo/15850
// [4] https://randallr.wordpress.com/2011/12/
// [5] https://medium.com/@fatlip/cuda-opengl-interop-e4edd8727c63
// [6] http://www.songho.ca/opengl/gl_pbo.html
//


#include "texture.h"

namespace cudademo
{

GLuint m_texOGL_3D;
GLuint m_texOGL_2D;                      // OpenGL 2D texture handle
GLuint m_PBO;                            // OpenGL PBO handle
struct cudaGraphicsResource* m_texCUDA;  // CUDA graphic resource (i.e., image)


 /*--------------------------------------------------------------------------------------------------------+
 |                                                 KERNELS                                                 |
 +--------------------------------------------------------------------------------------------------------*/

__global__
void simpleKernelTex(int _dimX, int _dimY, cudaTextureObject_t _texObject, unsigned char* _out)
{
    int nbChannels = 4; // RGBA

    int idX = blockIdx.x * blockDim.x * nbChannels + threadIdx.x * nbChannels;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;
    int strideX = blockDim.x * gridDim.x * nbChannels;
    int strideY = blockDim.y * gridDim.y;

    for (int i = idX; i < _dimX * nbChannels; i += strideX)
    {
        for (int j = idY; j < _dimY; j += strideY)
        {
            int index = j * _dimX * nbChannels + i;

            uchar4 p = tex2D<uchar4>(_texObject, i / nbChannels, j);

            // inverts RGB values
            _out[index] = 255 - p.x;
            _out[index + 1] = 255 - p.y;
            _out[index + 2] = 255 - p.z;
            _out[index + 3] = 255;
        }
    }

}


__global__
void simpleKernelPBO(int _dimX, int _dimY, unsigned char* _in, unsigned char* _out)
{
    int nbChannels = 4; // RGBA

    int idX = blockIdx.x * blockDim.x * nbChannels + threadIdx.x * nbChannels;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;
    int strideX = blockDim.x * gridDim.x * nbChannels;
    int strideY = blockDim.y * gridDim.y;

    for (int i = idX; i < _dimX * nbChannels; i += strideX)
    {
        for (int j = idY; j < _dimY; j += strideY)
        {
            int index = j * _dimX * nbChannels + i;

            // switches RGB channels
            _out[index] = _in[index + 2] ;
            _out[index + 1] = _in[index + 1];
            _out[index + 2] = _in[index];
            _out[index + 3] = 255;
        }
    }
}



__global__
void simpleKernelPBO3D(int _dimX, int _dimY, int _dimZ,unsigned char* _in, unsigned char* _out)
{
    int nbChannels = 4; // RGBA

    int idX = blockIdx.x * blockDim.x * nbChannels + threadIdx.x * nbChannels;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;
    int idZ = blockIdx.z * blockDim.z + threadIdx.z;
    int strideX = blockDim.x * gridDim.x * nbChannels;
    int strideY = blockDim.y * gridDim.y;
    int strideZ = blockDim.z * gridDim.z;

    for (int i = idX; i < _dimX * nbChannels; i += strideX)
    {
        for (int j = idY; j < _dimY; j += strideY)
        {
            for (int k = idZ; k < _dimZ; k += strideZ)
            {
                int index =   (k * _dimY * _dimX * nbChannels) 
                            + (j * _dimX * nbChannels) 
                            + i;

                // simply writes uniform color
                _out[index] = 0;
                _out[index + 1] = 255;
                _out[index + 2] = 0;
                _out[index + 3] = 255;
            }
        }
    }
}


 /*--------------------------------------------------------------------------------------------------------+
 |                                               HOST CODE                                                 |
 +--------------------------------------------------------------------------------------------------------*/


/* 
 * Creates an empty OpenGL 3D texture
 */
void createTex3D(GLuint& _tex3D, const int _imWidth, const int _imHeight, const int _imDepth)
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
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8UI, _imWidth, _imHeight, _imDepth, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL /*data*/);
    }

    printErrors();
    glBindTexture(GL_TEXTURE_3D, 0);
}


/* 
 * Creates an OpenGL 2D texture
 */
void createTex2D(GLuint& _tex2D, const int _imWidth, const int _imHeight,
                 const std::vector<unsigned char>& _imgData)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glGenTextures(1, &_tex2D);
    glBindTexture(GL_TEXTURE_2D, _tex2D);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

        // int 8 rgba texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, _imWidth, _imHeight, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, &_imgData[0]);
    }

    printErrors();
    glBindTexture(GL_TEXTURE_2D, 0);
}


/* 
 * Creates an OpenGL PBO
 */
void createPBO(GLuint& _PBO, const int _imTotalDim, const std::vector<unsigned char>& _imgData )
{
    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &_PBO);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _PBO);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(unsigned char) * _imTotalDim, &_imgData[0], GL_STREAM_DRAW);

    printErrors();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}


/* 
 * TextureObject test
 * Loads an image into an OpenGL texture, modify its content with CUDA, 
 * then export the OpenGL texture to a new file
 */
void testTextureObject(const std::string& _filenameIn, const std::string& _filenameOut)
{
    // Read PNG 
    std::vector<unsigned char> imData;
    unsigned imWidth, imHeight;
    unsigned lodepngErr = lodepng::decode(imData, imWidth, imHeight, _filenameIn);
    if (lodepngErr != 0) { std::cerr << "lodepng::decode(): " << lodepng_error_text(lodepngErr) << std::endl;
                           std::exit(EXIT_FAILURE); }
    // total nb of 8b values in the image (i.e., nb of pixels * nb of channels)
    unsigned int imTotalDim = imWidth * imHeight * 4;

    // Init OpenGL texture with PNG image
    cudademo::createTex2D(m_texOGL_2D, imWidth, imHeight, imData);


    // CUDA / OpenGL interop -------------------------------------------------------

    // 1. Register the OpenGL texture as a CUDA graphics resource ------------------
    cudaError_t err =  cudaGraphicsGLRegisterImage( &m_texCUDA, 
                                                    m_texOGL_2D, 
                                                    GL_TEXTURE_2D,
                                                    cudaGraphicsMapFlagsNone);
    printErrors(err);


    // 2. map the CUDA graphic resource to a CUDA array ----------------------------
    cudaArray *cudaArray;
    err = cudaGraphicsMapResources(1, &m_texCUDA, 0);
    printErrors(err);
    err = cudaGraphicsSubResourceGetMappedArray( &cudaArray, m_texCUDA, 0, 0);
    printErrors(err);


    // 3. create a cudaTextureObject -----------------------------------------------
    // 3.1. create a cudaResourceDescriptor
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cudaArray;
    // 3.2. create a cudaTextureDescriptor
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;
    // 3.3. Finally create a cudaTextureObject
    cudaTextureObject_t texObject;
    err = cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL);
    printErrors(err);


    // 4. Allocate a device memory buffer to store result --------------------------
    unsigned char *d_output;
    cudaMalloc((void**)&d_output, imTotalDim * sizeof(unsigned char));


    // 5. Execute kernel, using textureObject as input -----------------------------
    dim3 blockSize(16, 16);
    dim3 numBlocks((imWidth + blockSize.x - 1) / blockSize.x, 
                   (imHeight + blockSize.y - 1) / blockSize.y);
    simpleKernelTex<<<numBlocks, blockSize>>> (imWidth, imHeight, texObject, d_output);
    cudaDeviceSynchronize();
    std::vector<unsigned char> result;
    result.assign(imTotalDim, 100);
    cudaMemcpy(&result[0], d_output, imTotalDim * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // export final result,i.e., the data of original OpenGL texture, with kernel applied
    lodepngErr = lodepng::encode(_filenameOut, result, imWidth, imHeight);
    if (lodepngErr != 0) { std::cerr << "lodepng::encode(): " << lodepng_error_text(lodepngErr) << std::endl;
                           std::exit(EXIT_FAILURE); }


    // 6. copy result back to the CUDA array
    err = cudaMemcpy2DToArray(cudaArray, 0, 0, d_output,
                              sizeof(unsigned char) * imWidth * 4, 
                              sizeof(unsigned char) * imWidth * 4, imHeight,
                              cudaMemcpyDeviceToDevice);
    printErrors(err);
    cudaDeviceSynchronize();

    //export opengl texture to check that it contains transformed image now
    glBindTexture(GL_TEXTURE_2D, m_texOGL_2D);
    imData.clear();
    imData.assign(imTotalDim, 100);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, &imData[0]);
    printErrors();
    lodepngErr = lodepng::encode("../src/resOGLtex_after.png", imData, imWidth, imHeight);
    if (lodepngErr != 0) { std::cerr << "lodepng::encode(): " << lodepng_error_text(lodepngErr) << std::endl;
                           std::exit(EXIT_FAILURE); }

    glBindTexture(GL_TEXTURE_2D, 0);



    cudaFree(d_output);

    return;
}


/* 
 * PBO test
 * Pixel Buffer Object
 */
void testPBO(const std::string& _filenameIn, const std::string& _filenameOut)
{
    // Read PNG 
    std::vector<unsigned char> imData;
    unsigned imWidth, imHeight;
    unsigned lodepngErr = lodepng::decode(imData, imWidth, imHeight, _filenameIn);
    if (lodepngErr != 0) { std::cerr << "lodepng::decode(): " << lodepng_error_text(lodepngErr) << std::endl;
                           std::exit(EXIT_FAILURE); }
    // total nb of 8b values in the image (i.e., nb of pixels * nb of channels)
    unsigned int imTotalDim = imWidth * imHeight * 4;

    // Init OpenGL texture with PNG image
    cudademo::createTex2D(m_texOGL_2D, imWidth, imHeight, imData);

    // Init OpenGL PBO with PNG image
    cudademo::createPBO(m_PBO, imTotalDim, imData);


    // CUDA / OpenGL interop -------------------------------------------------------

    // 1. Register the PBO as a CUDA graphics resource -----------------------------
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&m_texCUDA, m_PBO, cudaGraphicsMapFlagsNone); 
    printErrors(err);


    // 2. map the CUDA graphic resource to a pointer -------------------------------
    unsigned char* cuda_mapped_ptr = nullptr;
    size_t numBytes = imTotalDim * sizeof(unsigned char);
    cudaMalloc((void**)&cuda_mapped_ptr, numBytes);
    err = cudaGraphicsMapResources(1, &m_texCUDA, 0);
    printErrors(err);
    size_t size;
    err = cudaGraphicsResourceGetMappedPointer( (void **)&cuda_mapped_ptr, &size, m_texCUDA);
    printErrors(err);


    // 3. Allocate a device memory buffer to store result --------------------------
    unsigned char *d_output;
    cudaMalloc((void**)&d_output, numBytes);


    // 4. Execute kernel, using CUDA mapped ptr as input ---------------------------
    dim3 blockSize(16, 16);
    dim3 numBlocks((imWidth + blockSize.x - 1) / blockSize.x, 
                   (imHeight + blockSize.y - 1) / blockSize.y);
    simpleKernelPBO<<<numBlocks, blockSize>>> (imWidth, imHeight, cuda_mapped_ptr, d_output);
    cudaDeviceSynchronize();

    // 5. Export result to a new image ---------------------------------------------
    std::vector<unsigned char> result;
    result.assign(imTotalDim, 100);
    cudaMemcpy(&result[0], d_output, numBytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    lodepngErr = lodepng::encode(_filenameOut, result, imWidth, imHeight);
    if (lodepngErr != 0) { std::cerr << "lodepng::encode(): " << lodepng_error_text(lodepngErr) << std::endl;
                           std::exit(EXIT_FAILURE); }


    // 6. Finally unmap resource and free memory -----------------------------------
    err = cudaGraphicsUnmapResources(1, &m_texCUDA, 0);
    printErrors(err);
    cudaFree(d_output);
    cudaFree(cuda_mapped_ptr);
}



/* 
 * PBO3D test
 * Pixel Buffer Object for 3D texture
 */
void testPBO3D()
{
    unsigned imWidth = 16, imHeight = 16, imDepth = 16;
    // total nb of 8b values in the image (i.e., nb of pixels * nb of channels)
    unsigned int imTotalDim = imWidth * imHeight * imDepth * 4;

    // Init OpenGL texture with PNG image
    cudademo::createTex3D(m_texOGL_2D, imWidth, imHeight, imDepth);

    // Init OpenGL PBO with PNG image
    std::vector<unsigned char> emptyVector;
    emptyVector.assign(imTotalDim, 0);
    cudademo::createPBO(m_PBO, imTotalDim, emptyVector);


    // CUDA / OpenGL interop -------------------------------------------------------

    // 1. Register the PBO as a CUDA graphics resource -----------------------------
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&m_texCUDA, m_PBO, cudaGraphicsMapFlagsNone); 
    printErrors(err);


    // 2. map the CUDA graphic resource to a pointer -------------------------------
    unsigned char* cuda_mapped_ptr = nullptr;
    size_t numBytes = imTotalDim * sizeof(unsigned char);
    cudaMalloc((void**)&cuda_mapped_ptr, numBytes);
    err = cudaGraphicsMapResources(1, &m_texCUDA, 0);
    printErrors(err);
    size_t size;
    err = cudaGraphicsResourceGetMappedPointer( (void **)&cuda_mapped_ptr, &size, m_texCUDA);
    printErrors(err);


    // 3. Allocate a device memory buffer to store result --------------------------
    unsigned char *d_output;
    cudaMalloc((void**)&d_output, numBytes);


    // 4. Execute kernel, using CUDA mapped ptr as input ---------------------------
    dim3 blockSize(8, 8, 8);
    dim3 numBlocks((imWidth + blockSize.x - 1) / blockSize.x, 
                   (imHeight + blockSize.y - 1) / blockSize.y,
                   (imDepth + blockSize.z - 1) / blockSize.z );
    simpleKernelPBO3D<<<numBlocks, blockSize>>> (imWidth, imHeight, imDepth, cuda_mapped_ptr, d_output);
    cudaDeviceSynchronize();

    // 5. Export a portion of result to a new image --------------------------------
    std::vector<unsigned char> result;
    result.assign(imWidth * imHeight * 4, 100);
    cudaMemcpy(&result[0], d_output, imWidth * imHeight * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    unsigned lodepngErr = lodepng::encode("../src/resPBO3D.png", result, imWidth, imHeight);
    if (lodepngErr != 0) { std::cerr << "lodepng::encode(): " << lodepng_error_text(lodepngErr) << std::endl;
                           std::exit(EXIT_FAILURE); }


    // 6. Finally unmap resource and free memory -----------------------------------
    err = cudaGraphicsUnmapResources(1, &m_texCUDA, 0);
    printErrors(err);
    cudaFree(d_output);
    cudaFree(cuda_mapped_ptr);
}

} // namespace cudademo