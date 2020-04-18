import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.python.framework import convert_to_constants 
import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

class VisionHandler():
    __window_name = None
    __frame = None
    __camera_mapX = None
    __camera_mapY = None
    __frame_undistorted = None
    __frame_resized = None
    __frame_gray = None
    __output_frame = None
    __input_dimensions = (1080, 1920) 
    __window_size = (480, 480)
    __model = tf.saved_model.load("cnn/trt_cnn_model.pb")
    __graph_func = __model.signatures["serving_default"]
    __frozen_graph = convert_to_constants.convert_variables_to_constants_v2(__graph_func)
    
    def __open_onboard_camera(self):
        return cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,format=UYVY,width=" + str(self.__input_dimensions[1]) + ",height=" + str(self.__input_dimensions[0]) + ", framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! nvvidconv ! video/x-raw,format=(string)BGRx ! videoconvert ! video/x-raw,format=(string)BGR ! appsink sync=0", cv2.CAP_GSTREAMER)
     
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
        
    def __predict_classes(self):
        frame_for_prediction = cv2.normalize(self.__frame_gray.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        frame_for_prediction = frame_for_prediction.reshape(1, 150, 150, 1)
        frame_for_prediction = tf.convert_to_tensor(frame_for_prediction)
        prediction_list = self.__frozen_graph(frame_for_prediction)[0].numpy()
        print(prediction_list)
        cv2.putText(self.__frame_gray, "Closed hand " + str(prediction_list.item(0)*100) + "%", (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(self.__frame_gray, "Open hand " + str(prediction_list.item(1)*100) + "%", (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
    
    def read_cameras(self):
        video_capture = self.__open_onboard_camera()
        
        if video_capture.isOpened():
            self.__prepare_window()
            
            return_value, self.__frame = video_capture.read()
            
            self.__prepare_undistortion()
            
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
                self.__frame_resized = cv2.resize(self.__frame_undistorted, (267, 150))
                self.__frame_gray = cv2.cvtColor(self.__frame_resized[:,58:208], cv2.COLOR_BGR2GRAY)
                
                self.__predict_classes()
                
                cv2.putText(self.__frame_gray, "FPS:" + str(fps), (90, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                cv2.imshow(self.__window_name, self.__frame_gray)
                
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
    print("Gesture Detection and Classification in Real-Time\n")
    print("OpenCV version: {}".format(cv2.__version__))
    vision_handler = VisionHandler()
    vision_handler.read_cameras()
