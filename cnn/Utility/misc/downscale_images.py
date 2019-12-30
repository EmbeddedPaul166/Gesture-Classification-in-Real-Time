import cv2
import numpy as np
import glob

path_v = sorted(glob.glob("../../images/training_set/victory/victory*.jpg"))
path_f = sorted(glob.glob("../../images/training_set/fist/fist*.jpg"))
path_o = sorted(glob.glob("../../images/training_set/open_hand/open_hand*.jpg")) 
path_h = sorted(glob.glob("../../images/training_set/horns/horns*.jpg"))
path_t = sorted(glob.glob("../../images/training_set/thumbup/thumbup*.jpg")) 

count = 1

for (image_path_v, image_path_f, image_path_o, image_path_h, image_path_t) in zip(path_v, path_f, path_o, path_h, path_t):
    if count > 2000:
        break
    
    image_v = cv2.imread(image_path_v)
    height_v, width_v, channels_v = image_v.shape
    #if height_v != 90 or width_v != 90:
    image_vr = cv2.resize(image_v, (90, 90))
    cv2.imwrite("../../images/training_set/victory/victory%d.jpg"%(count), image_vr)

    image_f = cv2.imread(image_path_f)
    height_f, width_f, channels_f = image_f.shape
    #if height_f != 90 or width_f != 90:
    image_fr = cv2.resize(image_f, (90, 90))
    cv2.imwrite("../../images/training_set/fist/fist%d.jpg"%(count), image_fr)

    image_o = cv2.imread(image_path_o)
    height_o, width_o, channels_o = image_o.shape
    #if height_o != 90 or width_o != 90:
    image_or = cv2.resize(image_o, (90, 90))
    cv2.imwrite("../../images/training_set/open_hand/open_hand%d.jpg"%(count), image_or)

    image_h = cv2.imread(image_path_h)
    height_h, width_h, channels_h = image_h.shape
    #if height_h != 90 or width_h != 90:
    image_hr = cv2.resize(image_h, (90, 90))
    cv2.imwrite("../../images/training_set/horns/horns%d.jpg"%(count), image_hr)
    
    image_t = cv2.imread(image_path_t)
    height_t, width_t, channels_t = image_t.shape
    #if height_t != 90 or width_t != 90:
    image_tr = cv2.resize(image_t, (90, 90))
    cv2.imwrite("../../images/training_set/thumbup/thumbup%d.jpg"%(count), image_tr)
    
    count += 1
        
