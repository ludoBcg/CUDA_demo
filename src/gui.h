/*********************************************************************************************************************
 *
 * gui.h
 * 
 * GUI widget
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#ifndef GUI_H
#define GUI_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "volume.h"
#include "tools.h"
#include "mesh.h"


namespace cudademo
{

struct UI 
{
    bool useTexNearest = false;       /*! flag to indicate if texture uses GL_NEAREST param (if not, uses GL_LINEAR by default)*/
    int sliceIdA;                     /*! ID of the Axial slice to visualize*/
    int sliceIdC;                     /*! ID of the Coronal slice to visualize*/
    int sliceIdS;                     /*! ID of the Sagittal slice to visualize*/
};


void GUI( UI& _ui,
          Volume& _volume,
          GLuint& _volTex,
          GLuint& _volPBO )
{

    // Always show GUI at top-left corner when starting
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);

    if (ImGui::Begin("Settings"))
    {
        // ImGui frame rate measurement
        float frameRate = ImGui::GetIO().Framerate;
        ImGui::Text("FrameRate: %.3f ms/frame (%.1f FPS)", 1000.0f / frameRate, frameRate);

        ImGui::Separator();
        ImGui::SliderInt("Z slice", &_ui.sliceIdA, 1, _volume.getDimensions()[2]);
        ImGui::SliderInt("Y slice", &_ui.sliceIdC, 1, _volume.getDimensions()[1]);
        ImGui::SliderInt("X slice", &_ui.sliceIdS, 1, _volume.getDimensions()[0]);

        if (ImGui::Checkbox("Show nearest voxel", &_ui.useTexNearest))
        {
            // re-build 3D texture
            build3DTexPBO(_volTex, _volPBO, _volume, _ui.useTexNearest);
        }

    } // end "Settings"

    ImGui::End();

    // render
    ImGui::Render();
}

} // namespace cudademo

#endif // GUI_H