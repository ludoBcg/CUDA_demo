/*********************************************************************************************************************
 *
 * tools.h
 *
 * Minimal classes for Camera
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#ifndef TOOLS_H
#define TOOLS_H

#include "volume.h"
#include "interop.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#define NOMINMAX // avoid min*max macros to interfer with std::min/max from <windows.h>

#include <glm/gtc/type_ptr.hpp>


namespace cudademo
{

/*!
* \class Camera
* \brief Handles camera matrices
*/
class Camera
{
    private:

        glm::mat4 m_projectionMatrix;     /*!< Perspective projection matrix */
        glm::mat4 m_viewMatrix;           /*!< View matrix */

        float m_nearPlane;                /*!< distance to near clip plane */
        float m_farPlane;                 /*!< distance to far clip plane */
        float m_fovy;                     /*!< field of view angle */
        float m_aspect;                   /*!< aspect ration */
        float m_zoomFactor;               /*!< factor applied to fov for zoom effect */
        

    public:


        /*!
        * \fn Camera
        * \brief Default constructor
        */
        Camera() : m_projectionMatrix(glm::mat4(1.0f)),
                   m_viewMatrix(glm::mat4(1.0f)),
                   m_nearPlane(0.1f),
                   m_farPlane(50.0f),
                   m_fovy(45.0f),
                   m_aspect(3.0f/4.0f),
                   m_zoomFactor(0.5f)
        {}


        /*!
        * \fn init
        * \brief Initialize camera attributes  and matrices
        *
        * \param _near : distance to near clip plane
        * \param _far : distance to far clip plane
        * \param _fov : field of view angle
        * \param _zoomFactor : factor applied to fov for zoom effect
        * \param _width : viewport width
        * \param _height : viewport height
        * \param _camCoords : 3D coords of the camera position
        * \param _centerCoords : 3D coords of the scene's center (i.e., the position to look at)
        * \param _radScene : radius of the scene (for orthogonal projection only)
        */
        void init(float _near, float _far, float _fov, float _zoomFactor, int _width, int _height, glm::vec3 _camCoords, glm::vec3 _centerCoords)
        {
            m_nearPlane = _near; 
            m_farPlane = _far;
            m_fovy = _fov;

            initProjectionMatrix(_width, _height, _zoomFactor);
            initViewMatrix(_camCoords, _centerCoords);
        }


        /*!
        * \fn initProjectionMatrix
        * \brief Initialize the perspective projection matrix given the viewport dimensions and zoom factor 
        */
        void initProjectionMatrix(int _width, int _height, float _zoomFactor)
        {
            m_aspect = (float)_width / (float)_height;
            m_zoomFactor = _zoomFactor;

            m_projectionMatrix = glm::perspective(glm::radians(m_fovy) * m_zoomFactor, m_aspect, m_nearPlane, m_farPlane);

        }


        /*!
        * \fn initViewMatrix
        * \brief Initialize view matrix given the 3D coords of the camera position and the scene's center (i.e., the position to look at)
        */
        void initViewMatrix(glm::vec3 _camCoords, glm::vec3 _centerCoords)
        {
            // define up direction vector
            glm::vec3 upVec = glm::vec3(0, 1, 0);
            // avoid up vector and cam position to be aligned
            if (_camCoords.x == 0 && _camCoords.z == 0)
                upVec = glm::vec3(0, 0, -1);

            m_viewMatrix = glm::lookAt(_camCoords, _centerCoords, upVec);
        }


        /*!
        * \fn getProjectionMatrix
        * \brief ProjectionMatrix getter
        */
        glm::mat4 getProjectionMatrix() { return m_projectionMatrix; }


        /*!
        * \fn getViewMatrix
        * \brief ViewMatrix getter
        */
        glm::mat4 getViewMatrix() { return m_viewMatrix; }


}; // end class Camera



/*!
* \fn readShaderSource
* \brief read shader program and copy it in a string
* \param _filename : shader file name
* \return string containing shader program
*/
std::string readShaderSource(const std::string& _filename)
{
    std::ifstream file(_filename);
    std::stringstream stream;
    stream << file.rdbuf();

    return stream.str();
}


/*!
* \fn showShaderInfoLog
* \brief print out shader info log (i.e. compilation errors)
* \param _shader : shader
*/
void showShaderInfoLog(GLuint _shader)
{
    GLint infoLogLength = 0;
    glGetShaderiv(_shader, GL_INFO_LOG_LENGTH, &infoLogLength);
    std::vector<char> infoLog(infoLogLength);
    glGetShaderInfoLog(_shader, infoLogLength, &infoLogLength, &infoLog[0]);
    std::string infoLogStr(infoLog.begin(), infoLog.end());
    std::cerr << "[SHADER INFOLOG] " << infoLogStr << std::endl;
}


/*!
* \fn showProgramInfoLog
* \brief print out program info log (i.e. linking errors)
* \param _program : program
*/
void showProgramInfoLog(GLuint _program)
{
    GLint infoLogLength = 0;
    glGetProgramiv(_program, GL_INFO_LOG_LENGTH, &infoLogLength);
    std::vector<char> infoLog(infoLogLength);
    glGetProgramInfoLog(_program, infoLogLength, &infoLogLength, &infoLog[0]);
    std::string infoLogStr(infoLog.begin(), infoLog.end());
    std::cerr << "[PROGRAM INFOLOG] " << infoLogStr << std::endl;
}


/*!
* \fn loadShaderProgram
* \brief load shader program from shader files
* \param _vertShaderFilename : vertex shader filename
* \param _fragShaderFilename : fragment shader filename
*/
GLuint loadShaderProgram(const std::string& _vertShaderFilename, const std::string& _fragShaderFilename, const std::string& _vertHeader="", const std::string& _fragHeader="")
{
    // read headers
    std::string vertHeaderSource, fragHeaderSource;
    vertHeaderSource = readShaderSource(_vertHeader);
    fragHeaderSource = readShaderSource(_fragHeader);


    // Load and compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    std::string vertexShaderSource = readShaderSource(_vertShaderFilename);
    if(!_vertHeader.empty() )
    {
        // if headers are provided, add them to the shader
        const char *vertSources[2] = {vertHeaderSource.c_str(), vertexShaderSource.c_str()};
        glShaderSource(vertexShader, 2, vertSources, nullptr);
    }
    else
    {
        // if no header provided, the shader is contained in a single file
        const char *vertexShaderSourcePtr = vertexShaderSource.c_str();
        glShaderSource(vertexShader, 1, &vertexShaderSourcePtr, nullptr);
    }
    glCompileShader(vertexShader);
    GLint success = 0;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) 
    {
        std::cerr << "[ERROR] loadShaderProgram(): Vertex shader compilation failed:" << std::endl;
        showShaderInfoLog(vertexShader);
        glDeleteShader(vertexShader);
        return 0;
    }


    // Load and compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    std::string fragmentShaderSource = readShaderSource(_fragShaderFilename);
    if(!_fragHeader.empty() )
    {
        // if headers are provided, add them to the shader
        const char *fragSources[2] = {fragHeaderSource.c_str(), fragmentShaderSource.c_str()};
        glShaderSource(fragmentShader, 2, fragSources, nullptr);
    }
    else
    {
        // if no header provided, the shader is contained in a single file
        const char *fragmentShaderSourcePtr = fragmentShaderSource.c_str();
        glShaderSource(fragmentShader, 1, &fragmentShaderSourcePtr, nullptr);
    }
    glCompileShader(fragmentShader);
    success = 0;
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) 
    {
        std::cerr << "[ERROR] loadShaderProgram(): Fragment shader compilation failed:" << std::endl;
        showShaderInfoLog(fragmentShader);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return 0;
    }


    // Create program object
    GLuint program = glCreateProgram();

    // Attach shaders to the program
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);


    // Link program
    glLinkProgram(program);

    // Check linking status
    success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) 
    {
        std::cerr << "[ERROR] loadShaderProgram(): Linking failed:" << std::endl;
        showProgramInfoLog(program);
        glDeleteProgram(program);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return 0;
    }

    // Clean up
    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);

    return program;
}


/*!
* \fn build3DTex
* \brief Create a 3D texture and copy volume data into it.
* \param _volTex : pointer to id of texture to generate
* \param _vol : 3D image data (i.e., volume)
* \param _useNearest : flag to indicate if texture uses GL_NEAREST param (if not, uses GL_LINEAR by default)
*/
void build3DTex(GLuint* _volTex, Volume* _vol, bool _useNearest)
{
    GLint param;
    _useNearest ? param = GL_NEAREST : param = GL_LINEAR;

    // generate 3D texture
    glGenTextures(1, _volTex);
    glBindTexture(GL_TEXTURE_3D, *_volTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, _vol->getDimensions().x, _vol->getDimensions().y, _vol->getDimensions().z, 0, GL_RED, GL_UNSIGNED_BYTE, _vol->getFront());
    
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, param);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, param);

    glBindTexture(GL_TEXTURE_3D, 0);
}


void buildPBO(GLuint& _volPBO, Volume& _vol, unsigned int _imByteSize)
{

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &_volPBO);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _volPBO);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, _imByteSize, _vol.getFront(), GL_STREAM_DRAW);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}


//https://stackoverflow.com/questions/25512354/opengl-cuda-interop-image-not-displaying-in-window
void build3DTexPBO(GLuint& _volTex, GLuint& _volPBO, Volume& _vol, bool _useNearest)
{
    GLint param;
    _useNearest ? param = GL_NEAREST : param = GL_LINEAR;

    // generate 3D texture
    glGenTextures(1, &_volTex);
    glBindTexture(GL_TEXTURE_3D, _volTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, _vol.getDimensions().x, _vol.getDimensions().y, _vol.getDimensions().z, 0, GL_RED, GL_UNSIGNED_BYTE, _vol.getFront());
    
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, param);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, param);

    glBindTexture(GL_TEXTURE_3D, 0);


    unsigned int imByteSize = sizeof(unsigned char) * _vol.getDimensions().x * _vol.getDimensions().y * _vol.getDimensions().z;
    buildPBO(_volPBO, _vol, imByteSize);
    cudademo::mapPBO(_volPBO, _volTex, imByteSize,
                     _vol.getDimensions().x, _vol.getDimensions().y, _vol.getDimensions().z);

}


} // namespace cudademo

#endif // TOOLS_H