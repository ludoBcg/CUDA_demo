/*********************************************************************************************************************
 *
 * volume.h
 *
 * Procedural 8b 3D image
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#ifndef VOLUME_H
#define VOLUME_H

#define NOMINMAX // avoid min*max macros to interfer with std::min/max


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


namespace cudademo
{

/*!
* \class Volume
* \brief A procedural 3D image with 8b data
*/
class Volume
{

    public:

        Volume() = default;

        virtual ~Volume() { m_data.clear(); }

        void volumeInit(glm::ivec3 _dims);

        inline uint8_t getValue1i(int _id) 
        {
            if( _id < 0 || _id >= m_data.size() )
                std::cerr << "[ERROR] Volume::getValue1i(): out of bound: " << _id << std::endl;

            return m_data[_id]; 
        }

        inline int getIdfromCoords(int _i, int _j, int _k) 
        { 
            if (_i < 0 || _i >= m_dimensions.x || _j < 0 || _j >= m_dimensions.y || _k < 0 || _k >= m_dimensions.z)
                std::cerr << "[ERROR] Volume::getIdfromCoords() : out of bound: " << _i << " " << _j << " " << _k << std::endl;

            return (m_dimensions.x * m_dimensions.y * _k) + (m_dimensions.x * _j) + _i; 
        }

        inline uint8_t getValue3i(int _i, int _j, int _k)
        {
            int index  = getIdfromCoords(_i, _j, _k);
            return getValue1i(index);
        }

        inline uint8_t getValue3ui(glm::ivec3 _uiCoords){ return getValue3i(_uiCoords.x, _uiCoords.y, _uiCoords.z); }


        inline void setValue3ui(unsigned int _i, unsigned int _j, unsigned int _k, uint8_t _val) { setValue1ui( getIdfromCoords(_i, _j, _k), _val); }
        inline void setValue3ui(glm::ivec3 _uiCoords, uint8_t _val) { setValue3ui(_uiCoords.x, _uiCoords.y, _uiCoords.z, _val); }
        inline void setValue1ui(unsigned int _id, uint8_t _val) { m_data[_id] = _val; }


        inline glm::ivec3 getDimensions() const { return m_dimensions; }
        inline glm::vec3 getOrigin() const { return m_origin; }
        inline glm::vec3 getSpacing() const { return m_spacing; }

        inline void setDimensions(glm::ivec3 _dimensions) { m_dimensions = _dimensions; }
        inline void setOrigin(glm::vec3 _origin) { m_origin = _origin; }
        inline void setSpacing(glm::vec3 _spacing) { m_spacing = _spacing; }

        inline uint8_t* getFront() { return &m_data[0]; }

        glm::mat4 volumeComputeScaleMatrix();

    protected:

        glm::ivec3 m_dimensions = { 0, 0, 0 };      /*!< volume dimensions (i.e. resolution) */
        glm::vec3 m_origin = { 0.0, 0.0, 0.0 };     /*!< volume origin (i.e. real coords of bottom corner in space) */
        glm::vec3 m_spacing = { 0.0, 0.0, 0.0 };    /*!< voxel spacing (i.e. real distance between two voxels along each axis) */
        std::vector<uint8_t> m_data;                /*!< voxel data (i.e. voxel grid) */

};

} // namespace cudademo

#endif // VOLUME_H