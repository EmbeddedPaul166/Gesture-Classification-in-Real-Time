#include "cuda_cv.hpp"

unsigned int height;
unsigned int width;

unsigned int heightAfterResize;
unsigned int widthAfterResize;

unsigned int finalHeight;
unsigned int finalWidth;

cv::cuda::GpuMat cudaFrame;
cv::cuda::GpuMat mog2_mask;

cv::Rect ROI;

cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(5, 5), 0, 0);
cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> pMOG2 = cv::cuda::createBackgroundSubtractorMOG2(1, 300, false);

void initialize_parameters(unsigned int inputHeight, unsigned int inputWidth,
                           unsigned int inputHeightAfterResize, unsigned int inputWidthAfterResize,
                           unsigned int inputFinalHeight, unsigned int inputFinalWidth)
{
    height = inputHeight;
    width = inputWidth;
    heightAfterResize = inputHeightAfterResize;
    widthAfterResize = inputWidthAfterResize;
    finalHeight = inputFinalHeight;
    finalWidth = inputFinalWidth;
    int shift = (inputWidthAfterResize - inputFinalWidth)/2;
    ROI = cv::Rect(shift, 0, inputFinalWidth, inputFinalHeight);
}

void extract_foreground(unsigned char * inputFrame, unsigned char * outputFrame)
{
    cv::Mat frameCPU(cv::Size(width, height), CV_8UC3, inputFrame); 
    cv::Mat outputFrameCPU(cv::Size(finalWidth, finalHeight), CV_8UC1, outputFrame); 
    
    cudaFrame.upload(frameCPU);    
    
    cv::cuda::resize(cudaFrame, cudaFrame, cv::Size(widthAfterResize, heightAfterResize));
    
    cudaFrame = cudaFrame(ROI);
    
    gaussianFilter -> apply(cudaFrame, cudaFrame);
    
    pMOG2 -> apply(cudaFrame, mog2_mask, 0);
    
    mog2_mask.download(outputFrameCPU);
}
