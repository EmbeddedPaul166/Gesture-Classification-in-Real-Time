# Gesture-Classification-in-Real-Time
This is an application written in Python and C++, utilizing OpenCV, TensorFlow and underlying CUDA computing power of Jetson TX2. Dedicated for Antmicro's TX2/TX2i Deep Learning Kit. It's goal is to classify gestures in real time. Classification is done via trained convolutional neural network. It differentiates between 5 gestures:
<img src="https://github.com/EmbeddedPaul166/Gesture-Classification-in-Real-Time/blob/master/images/example_output/victory.png" height="180" width="320">
<img src="https://github.com/EmbeddedPaul166/Gesture-Classification-in-Real-Time/blob/master/images/example_output/horns.png" height="180" width="320">
<img src="https://github.com/EmbeddedPaul166/Gesture-Classification-in-Real-Time/blob/master/images/example_output/open_hand.png" height="180" width="320">
<img src="https://github.com/EmbeddedPaul166/Gesture-Classification-in-Real-Time/blob/master/images/example_output/fist.png" height="180" width="320">
<img src="https://github.com/EmbeddedPaul166/Gesture-Classification-in-Real-Time/blob/master/images/example_output/thumbup.png" height="180" width="320">

### Hardware components:
- Antmicro TX2/TX2i Deep Learning Kit

### Software components:
- Python 3.5
- C++ 11
- OpenCV 3.4.1
- TensorFlow GPU 1.14
- CUDA 9.0
- CuDNN 7.1.5
- TensorRT 4.0.2

### Algorithm steps
1. Download frame from camera.
2. Undistort frame based on calibration output.
3. Upload frame to GPU.
4. Resize image to 80x45 and crop it to 45x45 resolution.
5. Apply Gaussian filter to decrease noise.
6. Subtract background using MOG2.
7. Download frame from GPU to CPU.
8. Predict gesture class using convolutional neural network.
9. If any gesture class probability exceeds 90%, print the gesture name on the screen.

### Optimization
Steps 1, 2, 8 and 9 are done in Python. Steps 3, 4, 5, 6, 7 are done in C++ via a dynamic library. The reason for this is that OpenCV currently doesn't contain Python CUDA API. Interfacing between Python and C++ was done using ctypes module. Unfortunately frames from the camera are downloaded in 1920x1080 and 30 FPS (Full HD). I wasn't able to lower this using v4l2src pipeline (according to documentation with 1280x720 I could achieve 60 FPS). Perhaps in the future I will find a way to tweak the pipeline to increase FPS.

### CNN architecture
1. Convolution 2D 32, 3x3, ReLU activation, no padding
2. Max pooling 2x2
3. Convolution 2D 64, 3x3, ReLU activation, no padding
4. Max pooling 2x2
5. Convolution 2D 128, 3x3, ReLU activation, no padding
6. Max pooling 2x2
7. Dropout 0.25
8. Flatten
9. Dense layer 128 neurons, ReLU activation
10. Dropout 0.5
11. Dense layer, 5 neurons, Softmax activation

### Benchmarking
Framerate fluctuates between 8-12 FPS. Sometimes, depending on the system load it may drop lower for a second.
Convolutional neural network was trained on a custom dataset of 20000 black and white 45x45 images with split of 70% 15% 15% for training, validation and test data. It achieved 96% accuracy on test dataset.
### Running
Before running you'll need to compile C++ dynamic library. To do that change directory to cpp_cuda and input:
```
make
```
After that go back to main directory and input:
```
python3 gesture_classification.py
```
