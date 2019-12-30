from tensorflow.keras.models import load_model
import cv2
import numpy as np
import ctypes as C
import time

class VisionHandler(): 
    cuda = C.CDLL("cpp_cuda/lib/cuda_cv.so")
   
    __window_name = None
    __frame = None
    __output_frame = None
    __camera_mapX = None
    __camera_mapY = None
    __frame_undistorted = None
    __input_dimensions = (1080, 1920) 
    __height_crop_range = (0, 1080)
    __width_crop_range = (420, 1500)
    __window_size = (720, 1280)
    __resize_dimensions = (45, 80) 
    __final_dimensions = (45, 45) 
    __input_number_of_channels = 3
    __output_number_of_channels = 1
    __frame_ptr = np.zeros(dtype = np.uint8, shape = (45, 45)).tostring()
    __model = load_model("cnn/cnn_model.h5")
    
    def __open_onboard_camera(self):
        return cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,format=UYVY,width=" + str(self.__input_dimensions[1]) + ",height=" + str(self.__input_dimensions[0]) + ", framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! nvvidconv ! video/x-raw,format=(string)BGRx ! videoconvert ! video/x-raw,format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
     
    def __is_window_closed(self):
        if cv2.getWindowProperty(self.__window_name, 0) < 0:
            return 1
        else:
            return 0
        
    def __prepare_window(self):
        self.__window_name = "Gesture Classification"
        cv2.namedWindow(self.__window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.__window_name, self.__window_size[1], self.__window_size[0])
        cv2.moveWindow(self.__window_name, 0, 0)
        cv2.setWindowTitle(self.__window_name, "Gesture Classification")
        
    def __prepare_undistortion(self):
        file_storage = np.load("camera_parameters/cameraCalibResults.npz")
        
        calibration_rms_error = file_storage["RMS_ERROR"];
        
        print("Calibration RMS error: ", calibration_rms_error, "\n")
        
        mat = file_storage["MAT"]
        dist = file_storage["DIST"]
            
        h, w = self.__frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mat, dist, (w, h), 0, (w, h))
        self.__camera_mapX, self.__camera_mapY = cv2.initUndistortRectifyMap(
                                     mat,
                                     dist,
                                     None,
                                     new_camera_matrix,
                                     (w, h),
                                     cv2.CV_32FC1)
        
    def __extract_foreground(self):
        frame_undistorted = np.fromstring(self.__frame_undistorted, np.uint8)                        
        frame_undistorted = np.reshape(frame_undistorted, (self.__input_dimensions[0], self.__input_dimensions[1], self.__input_number_of_channels))
        frame_undistorted = frame_undistorted.tostring()
        
        self.cuda.extract_foreground(frame_undistorted, self.__frame_ptr)
        
        self.__output_frame = np.fromstring(self.__frame_ptr, np.uint8)
        self.__output_frame = np.reshape(self.__output_frame, (self.__final_dimensions[0], self.__final_dimensions[1], self.__output_number_of_channels))
        
    def predict_classes(self):
        output_frame = np.expand_dims(self.__output_frame, axis=0)
        prediction_list = self.__model.predict_proba(output_frame)
        self.__frame_undistorted = self.__frame_undistorted[self.__height_crop_range[0]:self.__height_crop_range[1], self.__width_crop_range[0]:self.__width_crop_range[1]]
        
        if prediction_list.item(0) > 0.90:
            cv2.putText(self.__frame_undistorted, "Fist", (10,1000), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
        elif prediction_list.item(1) > 0.90:
            cv2.putText(self.__frame_undistorted, "Horns", (10,1000), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
        
        elif prediction_list.item(2) > 0.90:
            cv2.putText(self.__frame_undistorted, "Open hand", (10,1000), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
        
        elif prediction_list.item(3) > 0.90:
            cv2.putText(self.__frame_undistorted, "Thumb up", (10,1000), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
        
        elif prediction_list.item(4) > 0.90:
            cv2.putText(self.__frame_undistorted, "Victory", (10,1000), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
    
    
    def read_cameras(self):
        video_capture = self.__open_onboard_camera()
        
        if video_capture.isOpened():
            self.__prepare_window()
            
            return_value, self.__frame = video_capture.read()
            
            self.__prepare_undistortion()
            
            self.cuda.initialize_parameters(self.__input_dimensions[0], self.__input_dimensions[1],
                                            self.__resize_dimensions[0], self.__resize_dimensions[1],
                                            self.__final_dimensions[0], self.__final_dimensions[1])
            
            fps = 0
            frame_counter = 0
            start = time.time()
            
            while True:
                if self.__is_window_closed():
                    break
                
                return_value, self.__frame = video_capture.read()
                
                if return_value == False:
                    break
                
                self.__frame_undistorted = cv2.remap(self.__frame, self.__camera_mapX, self.__camera_mapY, cv2.INTER_LINEAR)
                
                self.__extract_foreground()
                
                self.predict_classes()
                
                cv2.putText(self.__frame_undistorted, "FPS:" + str(fps), (750,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
                
                cv2.imshow(self.__window_name, self.__frame_undistorted)
                
                frame_counter += 1
                
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                
                end = time.time()
                if (end - start) >= 1:
                   fps = frame_counter
                   frame_counter = 0
                   start = end
                
            video_capture.release()
            
            cv2.destroyAllWindows()
        else:
            print("Failed to open cameras")

if __name__ == "__main__": 
    print("Gesture Classification\n")
    print("OpenCV version: {}".format(cv2.__version__))
    vision_handler = VisionHandler()
    vision_handler.read_cameras()
