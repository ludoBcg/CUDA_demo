/*********************************************************************************************************************
 *
 * comparison.h
 *
 * CUDA_demo
 * Ludovic Blache
 *
 *********************************************************************************************************************/


#ifndef COMPARISON_H_ 
#define COMPARISON_H_ 

#include "utils.h"


namespace cudademo
{
    void saxpyCPU(float _a);

    void convolutionCPU(const std::string& _filenameIn, const std::string& _filenameOut);

} // namespace cudademo

#endif // COMPARISON_H_ 