/*********************************************************************************************************************
 *
 * mesh.h
 *
 * Buffer manager for mesh rendering
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#ifndef MESH_H
#define MESH_H

#define QT_NO_OPENGL_ES_2
#include <GL/glew.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

// Include GLM
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>


namespace cudademo
{

// The attribute locations used in the vertex shader
enum AttributeLocation 
{
    POSITION = 0,
    TEX3D = 1
};


/*!
* \class Mesh
* \brief Mesh datastructure with rendering functionalities
*/
class Mesh
{
    public:

        /*!
        * \fn Mesh
        * \brief Default constructor of Mesh
        */
        Mesh();

        /*!
        * \fn Mesh
        * \brief Constructor of Mesh
        */
        Mesh(unsigned int _orientation);

        /*!
        * \fn ~Mesh
        * \brief Destructor of Mesh
        */
        virtual ~Mesh();

        /*!
        * \fn drawSlice
        * \brief Draw slice-quad with 3D texture mapping
        * \param _program : shader program
        * \param _modelMat : model matrix
        * \param _viewMat :camera view matrix
        * \param _projMat :camera projection matrix
        * \param _tex3dMat :transformation to apply of tex coords (e.g., translation for slice scrolling)
        * \param _3dTex : 3D texture with volume data
        */
        void drawSlice(GLuint _program, glm::mat4 _modelMat, glm::mat4 _viewMat, glm::mat4 _projMat, glm::mat4 _tex3dMat, GLuint _3dTex);


    protected:

        GLuint m_meshVAO;           /*!< mesh VAO */
        GLuint m_defaultVAO;        /*!< default VAO */

        GLuint m_vertexVBO;         /*!< name of vertex 3D coords VBO */
        GLuint m_tex3dVBO;          /*!< name of 3D TexCoords coords VBO */
        GLuint m_indexVBO;          /*!< name of index VBO */

        unsigned int m_numVertices; /*!< number of vertices in the VBOs */
        unsigned int m_numIndices;  /*!< number of indices in the index VBO */


        /*!
        * \fn createSliceVAO
        * \brief Create slices (i.e., 3 quads) VAO and VBOs
        * Slices are unit-quads (centered on origin) to be scaled to match Volume's dimension
        * \param _orientation : defines if the quad to build is an axial slice (=0), coronal (=1), or sagital (2)
        */
        void createSliceVAO(unsigned int _orientation);
};

} // namespace cudademo

#endif // MESH_H