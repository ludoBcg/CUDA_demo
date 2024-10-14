/*********************************************************************************************************************
 *
 * main.cpp
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


// Standard includes 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <tchar.h>
#include "aclapi.h"

#include "basics.h"
#include "convolution.h"
#include "texture.h"
#include "comparison.h"

#include "gui.h"



// Window
GLFWwindow *m_window;
int m_winWidth = 1024;
int m_winHeight = 720;

// Light and cameras 
GLtools::Camera m_camera;
float m_zoomFactor;

// 3D objects
cudademo::Mesh* m_drawSliceZ;
cudademo::Mesh* m_drawSliceY;
cudademo::Mesh* m_drawSliceX;

glm::mat4 m_modelMatrix;        /*!<  model matrix of the mesh */
GLuint m_defaultVAO;            /*!<  default VAO */

std::shared_ptr<cudademo::Volume> m_volume;

// Textures
GLuint m_volTex;                /*!< Volume 3D texture */
GLuint m_volPBO;                /*!< Volume PBO for CUDA-OpenGL interoperability */

// shader programs
GLuint m_programSlice;          /*!< handle of the program object (i.e. shaders) for slice rendering */

// Slice orientation
enum sliceOrient { Z = 0, Y = 1, X = 2 };

// UI flags
cudademo::UI m_ui;

glm::vec2 m_prevMousePos(0.0f);

std::string shaderDir = "../../src/shaders/";   /*!< relative path to shaders folder  */


// Function declarations
void initialize();
void initScene();
void setupImgui(GLFWwindow *window);
void display();
void resizeCallback(GLFWwindow* window, int width, int height);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void runGUI();
int main(int argc, char** argv);




    /*------------------------------------------------------------------------------------------------------------+
    |                                                      INIT                                                   |
    +-------------------------------------------------------------------------------------------------------------*/


void initialize()
{   
    // init scene parameters
    m_zoomFactor = 1.0f;

    // Setup background color
    glClearColor(0.15f, 0.15f, 0.15f, 0.0f);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
        
    // init model matrix
    m_modelMatrix = glm::mat4(1.0f);

    // new 3D image
    m_volume = std::make_shared<cudademo::Volume>();
    m_volume->volumeInit(glm::ivec3(256,256,256));

    initScene();

    m_drawSliceZ = new cudademo::Mesh(Z);
    m_drawSliceY = new cudademo::Mesh(Y);
    m_drawSliceX = new cudademo::Mesh(X);

    // init shaders
    m_programSlice = cudademo::loadShaderProgram(shaderDir + "sliceTex.vert", shaderDir + "sliceTex.frag");                   // Render textured slices 


    // build 3D texture from volume and FBO for raycasting
    build3DTexPBO(m_volTex, m_volPBO, *m_volume, false);
    //build3DTex(&m_volTex, m_volume.get(), false);

    //// build PBO
    //unsigned int imByteSize = sizeof(unsigned char) * m_volume->getDimensions().x * m_volume->getDimensions().y * m_volume->getDimensions().z;
    //buildPBO(m_volPBO, *m_volume, imByteSize);
    //cudademo::mapPBO(m_volPBO, m_volTex, imByteSize,
    //                 m_volume->getDimensions().x, m_volume->getDimensions().y, m_volume->getDimensions().z);

}


void initScene()
{
    // init cameras
    m_camera.init(0.01f, 4.0f, 45.0f, 1.0f,
                  m_winWidth, m_winHeight,
                  glm::vec3(0.6f, 0.3f, 1.5f),
                  glm::vec3(0.0f, 0.0f, 0.0f), 0 );

    m_ui.sliceIdA = m_volume->getDimensions()[2] / 2;
    m_ui.sliceIdC = m_volume->getDimensions()[1] / 2;
    m_ui.sliceIdS = m_volume->getDimensions()[0] / 2;
}


void setupImgui(GLFWwindow *window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Scale font for hdpi screen 
    ImFontConfig cfg;
    ImGui::GetIO().Fonts->AddFontDefault(&cfg)->Scale = 1.0f;
    
    ImGui::StyleColorsDark();

    // platform and renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
}


    /*------------------------------------------------------------------------------------------------------------+
    |                                                     DISPLAY                                                 |
    +-------------------------------------------------------------------------------------------------------------*/

void display()
{
    // Bind default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Clear window with background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // resize viewport to window dimensions
    glViewport(0, 0, m_winWidth, m_winHeight);

    // get matrices
    glm::mat4 modelMat = m_modelMatrix * m_volume->volumeComputeScaleMatrix();
    glm::mat4 viewMat = m_camera.getViewMatrix();
    glm::mat4 projMat = m_camera.getProjectionMatrix();

    glEnable(GL_DEPTH_TEST);

    float translA = (float)m_ui.sliceIdA / (float)m_volume->getDimensions()[2];
    glm::mat4 translMatA = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 0.0f, 1.0f - translA));
    translA -= 0.5f;

    float translC = (float)m_ui.sliceIdC / (float)m_volume->getDimensions()[1];
    glm::mat4 translMatC = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 1.0f - translC, 0.0f));
    translC -= 0.5f;

    float translS = (float)m_ui.sliceIdS / (float)m_volume->getDimensions()[0];
    glm::mat4 translMatS = glm::translate(glm::mat4(1.0), glm::vec3(1.0f - translS, 0.0f, 0.0f));
    translS -= 0.5f;

    m_drawSliceZ->drawSlice(m_programSlice, modelMat * glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 0.0f, translA)), viewMat, projMat, translMatA, m_volTex);
    m_drawSliceY->drawSlice(m_programSlice, modelMat * glm::translate(glm::mat4(1.0), glm::vec3(0.0f, translC, 0.0f)), viewMat, projMat, translMatC, m_volTex);
    m_drawSliceX->drawSlice(m_programSlice, modelMat * glm::translate(glm::mat4(1.0), glm::vec3(translS, 0.0f, 0.0f)), viewMat, projMat, translMatS, m_volTex);

}


    /*------------------------------------------------------------------------------------------------------------+
    |                                                CALLBACK METHODS                                             |
    +-------------------------------------------------------------------------------------------------------------*/


void resizeCallback(GLFWwindow* window, int width, int height)
{
    m_winWidth = width;
    m_winHeight = height;

    // re-init camera
    m_camera.initProjectionMatrix(m_winWidth, m_winHeight, m_zoomFactor, 0);

    // keep drawing while resize
    display();

    // Swap between front and back buffer
    glfwSwapBuffers(m_window);
}


void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureKeyboard) { return; }  // Skip other handling

    if (key == GLFW_KEY_R && ( action == GLFW_PRESS || action == GLFW_REPEAT ) )
    {
        // rotate model
        m_modelMatrix = glm::rotate(m_modelMatrix, glm::radians(5.0f), glm::vec3(0.0f, -1.0f, 0.0f));
    }
    else if (key == GLFW_KEY_S && action == GLFW_PRESS)
    {
        // recompile shaders
        m_programSlice = cudademo::loadShaderProgram(shaderDir + "sliceTex.vert", shaderDir + "sliceTex.frag");
    }

}


void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (ImGui::GetIO().WantCaptureMouse) { return; }  // Skip other handling   

    // get mouse cursor position
    double x, y;
    glfwGetCursorPos(window, &x, &y);

    // update zoom factor
    double newZoom = m_zoomFactor - yoffset / 10.0f;
    if (newZoom > 0.0f && newZoom < 2.0f)
    {
        m_zoomFactor -= (float)yoffset / 10.0f;
        // update camera
        m_camera.initProjectionMatrix(m_winWidth, m_winHeight, m_zoomFactor, 0);
    }
    
}



    /*------------------------------------------------------------------------------------------------------------+
    |                                                      MAIN                                                   |
    +-------------------------------------------------------------------------------------------------------------*/

void runGUI()
{
    GUI(m_ui, *m_volume, m_volTex, m_volPBO);
}

int main(int argc, char** argv)
{
    std::cout << std::endl << "Log:" << std::endl;

    /* Initialize GLFW and create a window */
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // <-- activate this line on MacOS
    //glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE); // disable double buffering
    m_window = glfwCreateWindow(m_winWidth, m_winHeight, "CUDA demo", nullptr, nullptr);
    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, resizeCallback);
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetScrollCallback(m_window, scrollCallback);

    // init ImGUI
    setupImgui(m_window);


    // init GL extension wrangler
    glewExperimental = true;
    GLenum res = glewInit();
    if (res != GLEW_OK) 
    {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return 1;
    }
    
    std::cout << std::endl
              << "OpenGL version: " << glGetString(GL_VERSION) << std::endl
              << "Vendor: " << glGetString(GL_VENDOR) << std::endl;


    // 1. Basic kernels ---------------------------------------------------------
    {
        std::cout << "\n\n 1. Test basic kernels -------------------------\n";
        cudademo::testBasics();
        cudademo::saxpyCPU(2.0f);
    }

    // 2. 2D convolution kernel -------------------------------------------------
    {
        std::cout << "\n\n 2. Test 2D convolution ------------------------\n";
        cudademo::testConvolution("../../src/lenna.png", "../../src/resConvGPU.png");
        cudademo::convolutionCPU("../../src/lenna.png", "../../src/resConvCPU.png");
    }

    // 3. OpenGL Interoperability -----------------------------------------------
    {
        std::cout << "\n\n 3. Test CUDA / OpenGL interoperability --------\n";
        cudademo::testTextureObject("../../src/lenna.png", "../../src/resCUDAtex.png");
        cudademo::testPBO("../../src/lenna.png", "../../src/resPBO.png");
        cudademo::testPBO3D();
    }

    std::cout << "\n\n 4. Test PBO 3D -------------------------\n";

    glGenVertexArrays(1, &m_defaultVAO);
    glBindVertexArray(m_defaultVAO);

    // call init function
    initialize();

    // main rendering loop
    while (!glfwWindowShouldClose(m_window)) 
    {
        // process events
        glfwPollEvents();
        // start frame for ImGUI
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        // build GUI
        runGUI();

        // rendering
        display();
        
        // render GUI
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        // Swap between front and back buffer
        glfwSwapBuffers(m_window);
        //glFlush(); // when double buffering disabled
    }


    // Cleanup imGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Close window
    glfwDestroyWindow(m_window);
    glfwTerminate();

    return 0;
}
