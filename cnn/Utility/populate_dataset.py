import sys
import cv2
import numpy as np
import ctypes as C
import os

class VisionHandler(): 
    cuda = C.CDLL("../../cpp_cuda/lib/cuda_cv.so")
    
    __window_name = None
    __frame = None
    __output_frame = None
    __camera_mapX = None
    __camera_mapY = None
    __frame_undistorted = None
    __input_dimensions = (1080, 1920) 
    __resize_dimensions = (45, 80) 
    __final_dimensions = (45, 45) 
    __input_number_of_channels = 3
    __output_number_of_channels = 1
    __frame_ptr = np.zeros(dtype = np.uint8, shape = (45, 45)).tostring()
    __path_victory = "../../images/test_set/victory/"
    __name_victory = "victory"
    __path_fist = "../../images/test_set/fist/"
    __name_fist = "fist"
    __path_open_hand = "../../images/test_set/open_hand/"
    __name_open_hand = "open_hand"
    __path_horns = "../../images/test_set/horns/"
    __name_horns = "horns"
    __path_thumbup = "../../images/test_set/thumbup/"
    __name_thumbup = "thumbup"
    
    
    def __open_onboard_camera(self):
        return cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,format=UYVY,width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw,format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
     
    def __is_window_closed(self):
        if cv2.getWindowProperty(self.__window_name, 0) < 0:
            return 1
        else:
            return 0
        
    def __prepare_window(self):
        self.__window_name = "Gesture Recognition"
        cv2.namedWindow(self.__window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.__window_name, 1280, 720)
        cv2.moveWindow(self.__window_name, 0, 0)
        cv2.setWindowTitle(self.__window_name, "Gesture Recognition")
        
        try:
            if not os.path.exists(self.__path_victory):
                os.makedirs(self.__path_victory)
                self.__path_victory = os.path.dirname(self.__path_victory)
                try:
                    os.stat(self.__path_victory)
                except:
                    os.mkdir(self.__path_victory)
                    
            if not os.path.exists(self.__path_fist):
                os.makedirs(self.__path_fist)
                self.__path_fist = os.path.dirname(self.__path_fist)
                try:
                    os.stat(self.__path_fist)
                except:
                    os.mkdir(self.__path_fist)
        
            if not os.path.exists(self.__path_open_hand):
                os.makedirs(self.__path_open_hand)
                self.__path_open_hand = os.path.dirname(self.__path_open_hand)
                try:
                    os.stat(self.__path_open_hand)
                except:
                    os.mkdir(self.__path_open_hand)
        except:
            pass

    
    def __prepare_undistortion(self):
        file_storage = np.load("../../camera_parameters/cameraCalibResults.npz")
        
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
        self.__frame_undistorted = np.fromstring(self.__frame_undistorted, np.uint8)                        
        self.__frame_undistorted = np.reshape(self.__frame_undistorted, (self.__input_dimensions[0], self.__input_dimensions[1], self.__input_number_of_channels))
        self.__frame_undistorted = self.__frame_undistorted.tostring()
        
        self.cuda.extract_foreground(self.__frame_undistorted, self.__frame_ptr)
        
        self.__output_frame = np.fromstring(self.__frame_ptr, np.uint8)
        self.__output_frame = np.reshape(self.__output_frame, (self.__final_dimensions[0], self.__final_dimensions[1], self.__output_number_of_channels))
        
    def read_cameras(self):
        video_capture = self.__open_onboard_camera()
        
        if video_capture.isOpened():
            self.__prepare_window()
            
            return_value, self.__frame = video_capture.read()
            
            self.__prepare_undistortion()
            
            self.cuda.initialize_parameters(self.__input_dimensions[0], self.__input_dimensions[1],
                                            self.__resize_dimensions[0], self.__resize_dimensions[1],
                                            self.__final_dimensions[0], self.__final_dimensions[1])
            
            victory_count = 1
            fist_count = 1
            open_hand_count = 1
            horns_count = 1
            thumbup_count = 1
            
            while True: 
                if self.__is_window_closed():
                    break
                
                return_value, self.__frame = video_capture.read()
                
                if return_value == False:
                    break
                
                self.__frame_undistorted = cv2.remap(self.__frame, self.__camera_mapX, self.__camera_mapY, cv2.INTER_LINEAR)
                
                self.__extract_foreground()
                
                cv2.imshow(self.__window_name, self.__output_frame)
                
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("v"):
                    cv2.imwrite("%s%s%d.jpg"%(self.__path_victory, self.__name_victory, victory_count), self.__output_frame)
                    print("Saving victory image ", victory_count)
                    victory_count += 1
                    
                elif key == ord("f"):
                    cv2.imwrite("%s%s%d.jpg"%(self.__path_fist, self.__name_fist, fist_count), self.__output_frame)
                    print("Saving fist image ", fist_count)
                    fist_count += 1
                    
                elif key == ord("o"):
                    cv2.imwrite("%s%s%d.jpg"%(self.__path_open_hand, self.__name_open_hand, open_hand_count), self.__output_frame)
                    print("Saving open hand image ", open_hand_count)
                    open_hand_count += 1
                    
                elif key == ord("h"):
                    cv2.imwrite("%s%s%d.jpg"%(self.__path_horns, self.__name_horns, horns_count), self.__output_frame)
                    print("Saving horns image ", horns_count)
                    horns_count += 1
                
                elif key == ord("t"):
                    cv2.imwrite("%s%s%d.jpg"%(self.__path_thumbup, self.__name_thumbup, thumbup_count), self.__output_frame)
                    print("Saving thumbup image ", thumbup_count)
                    thumbup_count += 1
            
            video_capture.release()
            
            cv2.destroyAllWindows()
        else:
            print("Failed to open cameras")

if __name__ == '__main__': 
    print("Gesture Recognition\n")
    print("OpenCV version: {}".format(cv2.__version__))
    vision_handler = VisionHandler()
    vision_handler.read_cameras()
