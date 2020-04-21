# Gesture-Classification-in-Real-Time
This is a second version of this application. This time it's more computation heavy as I tried to tackle a problem of gesture classification in real time without background subtraction and in background agnostic way using custom CNN. Gestures to be classified:

<img src="https://github.com/EmbeddedPaul166/Gesture-Classification-in-Real-Time/blob/master/examples/closed_hand_example.png" height="320" width="320"> <img src="https://github.com/EmbeddedPaul166/Gesture-Classification-in-Real-Time/blob/master/examples/open_hand_example.png" height="320" width="320">

Base dataset is also a part of this repository as well as augmentation script, that produces a lot more data artifically via various techniques.

### Hardware components:
- Antmicro TX2/TX2i Deep Learning Kit

### Software components:
- Python 3.6
- OpenCV 4.1.1
- TensorFlow 2.1.0
- CUDA 10.0
- CuDNN 7.6.3
- TensorRT 6.0.1

### Algorithm steps
1. Download frame from camera.
2. Remap frame based on calibration output.
3. Resize frame to 267x150.
4. Cut 150x150 from it.
5. Convert it to grayscale.
6. Normalize for float values between 0 and 1
7. Reshape to 1x150x150x1 size
8. Convert to tensor
9. Predict using a frozen graph from TensorRT optimized model, previously trained in Tensorflow

### Optimization
Model is optimized using TensorRT which resulted in huge increase (4 -> 24) of FPS. It's due to Tensorflow using only about 10% GPU on TX2, despite information on the internet that it maps 100% of GPU on default. Conversion was done on saved model format, at the beginning of the main app converted saved model is loaded and converted to frozen graph with which inference is done.

### CNN architecture
For custom CNN architecture see https://github.com/EmbeddedPaul166/Gesture-Classification-in-Real-Time/blob/master/cnn/train_cnn_model.py

### Benchmarking
Framerate mostly stays at 24 FPS, where upper limit is 30 FPS due to the way camera driver is implemented.

### Running
```
python3 gesture_classification.py
```

### TensorRT technical note
I encountered extremely slow model loading with TensorRT. Later I found out it's due to incorrect protobuf installation. For more information how to fix it see: https://github.com/EmbeddedPaul166/Gesture-Classification-in-Real-Time/blob/master/util/note
