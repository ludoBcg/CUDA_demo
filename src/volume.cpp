/*********************************************************************************************************************
 *
 * volume.cpp
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#include <algorithm>

#include "volume.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/noise.hpp>


namespace cudademo
{

void Volume::volumeInit(glm::ivec3 _dims)
{
    // setup volume param
    m_dimensions = _dims;
    m_origin = glm::vec3(0, 0, 0);
    m_spacing = glm::vec3(0.5f, 0.5f, 0.5f);

    const int size = m_dimensions.x * m_dimensions.y * m_dimensions.z;
    m_data.resize(size);

    // Perlin noise
    #pragma omp parallel for
    for (int i = 0; i < m_dimensions.x; i++)
    {
        for (int j = 0; j < m_dimensions.y; j++)
        {
            for (int k = 0; k < m_dimensions.z; k++)
            {
                float val = glm::perlin(glm::vec3((float)i / (float)m_dimensions.x * 30.0, 
                                                  (float)j / (float)m_dimensions.y * 30.0,
                                                  (float)k / (float)m_dimensions.z * 30.0));

                // change range from [-1;1] to [0;255]
                unsigned char val8b = unsigned char((val + 1.0f) * 255.0f);

                setValue3ui(i, j, k, val8b);
            }
        }
    }
}


// Computes the model matrix for the volume image. This matrix can be
// used during rendering to scale a unit cube to the size of
// the volume image. Assumes that the min corner of the cube is centered at origin.
glm::mat4 Volume::volumeComputeScaleMatrix()
{
    // volume size
    glm::vec3 extent = glm::vec3(m_dimensions) * m_spacing;
    // scale to a diagonal of size 1 (so it fits in the screen)
    extent = glm::normalize(extent);

    //cube mesh size
    glm::vec3 extentC(1.0f, 1.0f, 1.0f);
    // compute factor to scale the cube to the appropriate volume size
    glm::vec3 scale = extent / extentC;

    // Assume cube min corner is centered on origin.
    return glm::scale(glm::mat4(1.0), scale);
}

} // namespace cudademo