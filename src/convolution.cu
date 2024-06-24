/*********************************************************************************************************************
 *
 * convolution.cu
 *
 * CUDA example code for 2D image convolution
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/

// Sources:
//
// [1] https://medium.com/@harsh20111997/cuda-programming-2d-convolution-8476300f566e
//


#include "convolution.h"

namespace cudademo
{

 /*--------------------------------------------------------------------------------------------------------+
 |                                                 KERNELS                                                 |
 +--------------------------------------------------------------------------------------------------------*/

__global__
void convol(int _dimX, int _dimY, unsigned char* _in, unsigned char* _out, float* _filter)
{
    int nbChannels = 4; // RGBA
    int winRadius = 1;  // filter window radius

    // Dimension, index, and stride along the X axis should always be multiplied by nb of channels 
    // (i.e., nb of actual unsigned char value stored for each pixel)

    int idX = blockIdx.x * blockDim.x * nbChannels + threadIdx.x * nbChannels;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = blockDim.x * gridDim.x * nbChannels;
    int strideY = blockDim.y * gridDim.y;

    for (int i = idX; i < _dimX * nbChannels; i += strideX)
    {
        for (int j = idY; j < _dimY; j += strideY)
        {
            // index of current thread in pixel grid
            int index = j * _dimX * nbChannels + i;

            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
            // 3x3 window scanning to apply filter
            for (int x = -winRadius * nbChannels, fx = 0; x <= winRadius * nbChannels; x += nbChannels, fx++)
            {
                for (int y = -winRadius, fy = 0; y <= winRadius; y++, fy++)
                {
                    if ((j + y) > 0 && (j + y) < _dimY && (i + x) > 0 && (i + x) < _dimX * 4)
                    {
                        // index of current neighbor in pixel grid
                        int indexNh = (j + y) * _dimX * nbChannels + (i + x);
                        // corresponding index in filter window
                        int indexWin = fy * 3 + fx;
                        // get filter value
                        float filterVal = _filter[indexWin];
                        // apply filter value to current pixel
                        sumR += (float)(_in[indexNh]) * filterVal;
                        sumG += (float)(_in[indexNh + 1]) * filterVal;
                        sumB += (float)(_in[indexNh + 2]) * filterVal;
                    }
                }
            }

            _out[index] = (unsigned char)sumR;
            _out[index + 1] = (unsigned char)sumG;
            _out[index + 2] = (unsigned char)sumB;
            _out[index + 3] = 255;
        }
    }

}



 /*--------------------------------------------------------------------------------------------------------+
 |                                               HOST CODE                                                 |
 +--------------------------------------------------------------------------------------------------------*/


void testConvolution(const std::string& _filenameIn, const std::string& _filenameOut)
{
    gpuTimer gpuChrono;

    // read PNG image and store pixel in vector of 8b values
    // (each pixels is represented by 4 consecutiove values: one for each RGBA channel)
    std::vector<unsigned char> data;
    unsigned width, height;
    unsigned error = lodepng::decode(data, width, height, _filenameIn);
    if (error != 0) 
    {
        std::cerr << "load2DTexture(): " << lodepng_error_text(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // total nb of 8b values in the image (i.e., nb of pixels * nb of channels)
    unsigned int dimTotal = width * height * 4;

    // normalized 3x3 Gaussian blur window
    float gaussFilter[9] = { 0.0625, 0.125, 0.0625,
                             0.125 , 0.25 , 0.125 ,
                             0.0625, 0.125, 0.0625 };

    // create 2 arrays to store original image (input) and result (output) on the device
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, dimTotal * sizeof(unsigned char));
    cudaMemcpy(d_input, &data[0], dimTotal * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output, dimTotal * sizeof(unsigned char));

    // copy filter on the device
    float *d_filter;
    cudaMalloc((void**)&d_filter, 9 * sizeof(float));
    cudaMemcpy(d_filter, &gaussFilter[0], 9 * sizeof(float), cudaMemcpyHostToDevice);

    // define 2D blocks of 16x16 threads
    dim3 blockSize(16, 16);
    dim3 numBlocks( (width + blockSize.x - 1) / blockSize.x, 
                   (height + blockSize.y - 1) / blockSize.y );

    gpuChrono.start("kernel convolution");

    convol<<<numBlocks, blockSize>>> (width, height, d_input, d_output, d_filter);

    gpuChrono.stop();

    cudaDeviceSynchronize();

    // write result into destination PNG file
    cudaMemcpy(&data[0], d_output, dimTotal * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    lodepng::encode(_filenameOut, data, width, height);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    return;
}

} // namespace cudademo