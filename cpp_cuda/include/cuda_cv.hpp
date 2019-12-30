#ifndef CPP_CUDA_CV_HPP
#define CPP_CUDA_CV_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>


extern "C" unsigned int height;
extern "C" unsigned int width;

extern "C" unsigned int heightAfterResize;
extern "C" unsigned int widthAfterResize;

extern "C" unsigned int finalHeight;
extern "C" unsigned int finalWidth;

extern "C" cv::cuda::GpuMat cudaFrame;
extern "C" cv::cuda::GpuMat mog2_mask;

extern "C" cv::Rect ROI;

extern "C" cv::Ptr<cv::cuda::Filter> gaussianFilter;
extern "C" cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> pMOG2;

extern "C" void initialize_parameters(unsigned int inputHeight, unsigned int inputWidth,
                                      unsigned int inputHeightAfterResize, unsigned int inputWidthAfterResize,
                                      unsigned int inputFinalHeight, unsigned int inputFinalWidth);

extern "C" void extract_foreground(unsigned char * inputFrame, unsigned char * outputFrame);

#endif /*CPP_CUDA_CV_HPP*/
