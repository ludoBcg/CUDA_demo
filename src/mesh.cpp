/*********************************************************************************************************************
 *
 * mesh.cpp
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#include "mesh.h"


namespace cudademo
{

Mesh::Mesh()
{
    createSliceVAO(0);
}


Mesh::Mesh(unsigned int _orientation)
{
    createSliceVAO(_orientation);
}


Mesh::~Mesh()
{
    glDeleteBuffers(1, &(m_vertexVBO));
    glDeleteBuffers(1, &(m_tex3dVBO));
    glDeleteBuffers(1, &(m_indexVBO));
    glDeleteVertexArrays(1, &(m_meshVAO));
}


void Mesh::createSliceVAO(unsigned int _orientation)
{

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> texcoords;

    if (_orientation == 0)
    {
        // Z-AXIS
        vertices = { glm::vec3(0.5f,  0.5f,  0.0f), glm::vec3(-0.5f,  0.5f,  0.0f), glm::vec3(-0.5f, -0.5f,  0.0f), glm::vec3(0.5f, -0.5f,  0.0f) };
        texcoords = { glm::vec3(0.0f, 0.0f, 0.0f),    glm::vec3(1.0f, 0.0f, 0.0f),   glm::vec3(1.0f, 1.0f, 0.0f),   glm::vec3(0.0f, 1.0f, 0.0f) };
    }
    else if (_orientation == 1)
    {
        // Y-AXIS
        vertices = { glm::vec3(0.5f,  0.0f,  0.5f), glm::vec3(-0.5f,  0.0f,  0.5f), glm::vec3(-0.5f,  0.0f, -0.5f), glm::vec3(0.5f,  0.0f, -0.5f) };
        texcoords = { glm::vec3(0.0f, 0.0f, 0.0f),    glm::vec3(1.0f, 0.0f, 0.0f),   glm::vec3(1.0f, 0.0f, 1.0f),   glm::vec3(0.0f, 0.0f, 1.0f) };
    }
    else if (_orientation == 2)
    {
        // X-AXIS
        vertices = { glm::vec3(0.0f,  0.5f,  0.5f), glm::vec3(0.0f,  0.5f, -0.5f), glm::vec3(0.0f, -0.5f, -0.5f), glm::vec3(0.0f, -0.5f,  0.5f) };
        texcoords = { glm::vec3(0.0f, 0.0f, 0.0f),   glm::vec3(0.0f, 0.0f, 1.0f),   glm::vec3(0.0f, 1.0f, 1.0f),   glm::vec3(0.0f, 1.0f, 0.0f) };
    }
    else
        std::cerr << "[ERROR] Mesh::createSliceVAO(): Invalide Slice orientation" << std::endl;

    std::vector<uint32_t> indices = { 0, 1, 2, 0, 2, 3 };


    // Generates and populates a VBO for vertex coords
    glGenBuffers(1, &(m_vertexVBO));
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBO);
    size_t verticesNBytes = vertices.size() * sizeof(vertices[0]);
    glBufferData(GL_ARRAY_BUFFER, verticesNBytes, vertices.data(), GL_STATIC_DRAW);

    // Generates and populates a VBO for 3D Tex coords
    glGenBuffers(1, &(m_tex3dVBO));
    glBindBuffer(GL_ARRAY_BUFFER, m_tex3dVBO);
    size_t texcoordsNBytes = texcoords.size() * sizeof(texcoords[0]);
    glBufferData(GL_ARRAY_BUFFER, texcoordsNBytes, texcoords.data(), GL_STATIC_DRAW);

    // Generates and populates a VBO for the element indices
    glGenBuffers(1, &(m_indexVBO));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexVBO);
    size_t indicesNBytes = indices.size() * sizeof(indices[0]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesNBytes, indices.data(), GL_STATIC_DRAW);


    // Creates a vertex array object (VAO) for drawing the mesh
    glGenVertexArrays(1, &(m_meshVAO));
    glBindVertexArray(m_meshVAO);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBO);
    glEnableVertexAttribArray(POSITION);
    glVertexAttribPointer(POSITION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, m_tex3dVBO);
    glEnableVertexAttribArray(TEX3D);
    glVertexAttribPointer(TEX3D, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexVBO);
    glBindVertexArray(m_defaultVAO); // unbinds the VAO

    // Additional information required by draw calls
    m_numVertices = (unsigned int)vertices.size();
    m_numIndices = (unsigned int)indices.size();

    // Clear temporary vectors
    vertices.clear();
    texcoords.clear();
    indices.clear();
}


void Mesh::drawSlice(GLuint _program, glm::mat4 _modelMat, glm::mat4 _viewMat, glm::mat4 _projMat, glm::mat4 _tex3dMat, GLuint _3dTex)
{
    // Activate program
    glUseProgram(_program);

    // bind textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, _3dTex);

    // Pass uniforms
    glUniformMatrix4fv(glGetUniformLocation(_program, "u_matM"), 1, GL_FALSE, &_modelMat[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(_program, "u_matV"), 1, GL_FALSE, &_viewMat[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(_program, "u_matP"), 1, GL_FALSE, &_projMat[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(_program, "u_matTex"), 1, GL_FALSE, &_tex3dMat[0][0]);
    glUniform1i(glGetUniformLocation(_program, "u_volumeTexture"), 0);

    // Draw!
    glBindVertexArray(m_meshVAO);                       // bind the VAO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexVBO);  // do not forget to bind the index buffer AFTER !

    glDrawElements(GL_TRIANGLES, m_numIndices, GL_UNSIGNED_INT, 0);

    glBindVertexArray(m_defaultVAO);

    glUseProgram(0);
}

} // namespace cudademo