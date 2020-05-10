import cv2
import numpy as np
import os
import sys

class VisionHandler(): 
    __window_name = None
    __frame = None
    __output_frame = None
    __camera_mapX = None
    __camera_mapY = None
    __frame_undistorted = None
    __frame_gray = None
    __frame_resized = None
    __frame_blurred = None
    __frame_rect = None
    __input_dimensions = (1080, 1920) 
    __window_size = (150, 150)
    
    def __open_onboard_camera(self):
        #return cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,format=UYVY,width=" + str(self.__input_dimensions[1]) + ",height=" + str(self.__input_dimensions[0]) + ", framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! nvvidconv ! video/x-raw,format=(string)BGRx ! videoconvert ! video/x-raw,format=(string)BGR ! appsink sync=0", cv2.CAP_GSTREAMER)
        return cv2.VideoCapture(0)
     
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
        file_storage = np.load("../camera_parameters/cameraCalibResults.npz")
        
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
        
    def read_cameras(self):
        video_capture = self.__open_onboard_camera()
        
        if video_capture.isOpened():
            self.__prepare_window()
            
            return_value, self.__frame = video_capture.read()
            
            self.__prepare_undistortion()
            
            #file_name_open = "training_set/open_hand/open_hand"
            #file_name_closed = "training_set/closed_hand/closed_hand"
            #file_name_open = "validation_set/open_hand/open_hand"
            #file_name_closed = "validation_set/closed_hand/closed_hand"
            file_name_open = "test_set/open_hand/open_hand"
            file_name_closed = "test_set/closed_hand/closed_hand"
            image_count_open = 1
            image_count_closed = 1
            
            while True:
                if self.__is_window_closed():
                    break
                
                return_value, self.__frame = video_capture.read()
                
                if return_value == False:
                    break
                
                self.__frame_undistorted = cv2.remap(self.__frame, self.__camera_mapX, self.__camera_mapY, cv2.INTER_LINEAR)
                self.__frame_resized = cv2.resize(self.__frame_undistorted, (267, 150))
                self.__frame_gray = cv2.cvtColor(self.__frame_resized[:,58:208], cv2.COLOR_BGR2GRAY)
                #self.__frame_gray = cv2.rotate(self.__frame_gray, cv2.ROTATE_90_CLOCKWISE)
                cv2.imshow(self.__window_name, self.__frame_gray)
                
                key = cv2.waitKey(1)
                
                if key == ord("q"):
                    break
                
                if key == ord("o"):
                    print("Saving image open hand ", image_count_open)
                    cv2.imwrite("%s%d.jpg"%(file_name_open, image_count_open), self.__frame_gray)
                    image_count_open+=1
                    
                if key == ord("c"):
                    print("Saving image closed hand", image_count_closed)
                    cv2.imwrite("%s%d.jpg"%(file_name_closed, image_count_closed), self.__frame_gray)
                    image_count_closed+=1
                
            video_capture.release()
            
            cv2.destroyAllWindows()
        else:
            print("Failed to open cameras")

if __name__ == "__main__": 
    print("Gesture Detection and Classification in Real-Time\n")
    print("OpenCV version: {}".format(cv2.__version__))
    vision_handler = VisionHandler()
    vision_handler.read_cameras()
