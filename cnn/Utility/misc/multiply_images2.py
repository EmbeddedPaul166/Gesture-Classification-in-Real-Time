#Script for image multiplication, edit parameters for your needs to multiply images to your needs
#For example I took 500 photos of each gesture for training set and by rotating and flipping I generated 1500 more, which left 2000 in total for all three gestures

import cv2
import numpy as np
import glob

path_h = sorted(glob.glob("../../images/training_set/horns/horns*.jpg"))   
path_t = sorted(glob.glob("../../images/training_set/thumbup/thumbup*.jpg")) 

r_count = 501
f_count = 1001
fr_count = 1501

for (image_path_h, image_path_t) in zip(path_h, path_t):
    if r_count > 1000 or f_count > 1500 or fr_count > 2000:
        break
    else:
        image_h = cv2.imread(image_path_h)
        image_hr = cv2.rotate(image_h, cv2.ROTATE_180)
        image_hf = cv2.flip(image_h, 0)
        image_hfr = cv2.rotate(image_hf, cv2.ROTATE_180)
        cv2.imwrite("../../images/training_set/horns/horns%d.jpg"%(r_count), image_hr)
        cv2.imwrite("../../images/training_set/horns/horns%d.jpg"%(f_count), image_hf)
        cv2.imwrite("../../images/training_set/horns/horns%d.jpg"%(fr_count), image_hfr)
        
        image_t = cv2.imread(image_path_t)
        image_tr = cv2.rotate(image_t, cv2.ROTATE_180)
        image_tf = cv2.flip(image_t, 0)
        image_tfr = cv2.rotate(image_tf, cv2.ROTATE_180)
        cv2.imwrite("../../images/training_set/thumbup/thumbup%d.jpg"%(r_count), image_tr)
        cv2.imwrite("../../images/training_set/thumbup/thumbup%d.jpg"%(f_count), image_tf)
        cv2.imwrite("../../images/training_set/thumbup/thumbup%d.jpg"%(fr_count), image_tfr)
        
        r_count += 1
        f_count += 1
        fr_count += 1
        
    
    
    
    
