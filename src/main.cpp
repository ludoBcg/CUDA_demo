/*********************************************************************************************************************
 *
 * main.cpp
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#include "basics.h"
#include "convolution.h"
#include "texture.h"
#include "comparison.h"


GLFWwindow *m_window;



 /*--------------------------------------------------------------------------------------------------------+
 |                                                  MAIN                                                   |
 +--------------------------------------------------------------------------------------------------------*/


int main(void)
{

    // 1. Basic kernels ---------------------------------------------------------
    {
        std::cout << "\n\n 1. Test basic kernels -------------------------\n";
        cudademo::testBasics();
        cudademo::saxpyCPU(2.0f);
    }

    // 2. 2D convolution kernel -------------------------------------------------
    {
        std::cout << "\n\n 2. Test 2D convolution ------------------------\n";
        cudademo::testConvolution("../src/lenna.png", "../src/resConvGPU.png");
        cudademo::convolutionCPU("../src/lenna.png", "../src/resConvCPU.png");
    }

    // 3. OpenGL Interoperability -----------------------------------------------
    {
        std::cout << "\n\n 3. Test CUDA / OpenGL interoperability --------\n";
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        m_window = glfwCreateWindow(600, 600, "CUDA demo", nullptr, nullptr);
        glfwMakeContextCurrent(m_window);

        // init GL extension wrangler
        glewExperimental = true;
        GLenum res = glewInit();
        if (res != GLEW_OK) {
            std::cerr << "GLEW init error: " << glewGetErrorString(res) << std::endl;
            return 1;
        }
        std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl
            << "Vendor: " << glGetString(GL_VENDOR) << std::endl;

        
        cudademo::testTextureObject("../src/lenna.png", "../src/resCUDAtex.png");

        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
            glfwSwapBuffers(m_window);
        }

        glfwDestroyWindow(m_window);
        glfwTerminate();
    }
    return 0;
}