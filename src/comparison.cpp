/*********************************************************************************************************************
 *
 * comparison.cpp
 *
 * CPU versions of algorithms for comparison
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/

#include "comparison.h"

namespace cudademo
{

void saxpyCPU(float _a)
{
    // Create X and Y arrays
    const int N = 1<<20;
    float *arrayX = new float[N];
    float *arrayY = new float[N];
    for (int i = 0; i < N; i++)
    {
        arrayX[i] = 1.0f;
        arrayY[i] = 2.0f;
    }

    GLtools::CpuTimer cpuChrono;
    cpuChrono.start("OpenMP saxpy");

    // parallelized loop using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        arrayY[i] = _a * arrayX[i] + arrayY[i];
    }
    
    cpuChrono.stop();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(arrayY[i] - 4.0f));
    std::cout << std::endl << "Max error: " << maxError << std::endl;

    delete [] arrayX;
    delete [] arrayY;

    return;
}


void convolutionCPU(const std::string& _filenameIn, const std::string& _filenameOut)
{
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

    std::vector<unsigned char> output;
    output.assign(dimTotal, 0);

    int nbChannels = 4; // RGBA
    int winRadius = 1;  // filter window radius

    GLtools::CpuTimer cpuChrono;
    cpuChrono.start("OpenMP convolution");

    // #pragma omp parallel for collapse(2) // OpenMP 2.0. not supported by msvc
    #pragma omp parallel for
    for (int i = 0; i < (int)(width * nbChannels); i += nbChannels)
    {
        for (int j = 0; j < (int)height; j ++)
        {
            // index of current thread in pixel grid
            int index = j * width * nbChannels + i;

            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
            // 3x3 window scanning to apply filter
            for (int x = -winRadius * nbChannels, fx = 0; x <= winRadius * nbChannels; x += nbChannels, fx++)
            {
                for (int y = -winRadius, fy = 0; y <= winRadius; y++, fy++)
                {
                    if ((j + y) > 0 && (j + y) < (int)height && (i + x) > 0 && (i + x) < (int)(width * 4))
                    {
                        // index of current neighbor in pixel grid
                        int indexNh = (j + y) * width * nbChannels + (i + x);
                        // corresponding index in filter window
                        int indexWin = fy * 3 + fx;
                        // get filter value
                        float filterVal = gaussFilter[indexWin];
                        // apply filter value to current pixel
                        sumR += (float)(data[indexNh]) * filterVal;
                        sumG += (float)(data[indexNh + 1]) * filterVal;
                        sumB += (float)(data[indexNh + 2]) * filterVal;
                    }
                }
            }

            output[index] = (unsigned char)sumR;
            output[index + 1] = (unsigned char)sumG;
            output[index + 2] = (unsigned char)sumB;
            output[index + 3] = 255;
        }
    }

    cpuChrono.stop();

    lodepng::encode(_filenameOut, output, width, height);

    return;
}

} // namespace cudademo