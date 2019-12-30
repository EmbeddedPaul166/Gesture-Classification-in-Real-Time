#Script for image multiplication by 90 degrees rotating and flipping, edit parameters for your needs to multiply images to your needs

import cv2
import numpy as np
import glob

path_v = sorted(glob.glob("../../images/test_set/victory/victory*.jpg"))   
path_f = sorted(glob.glob("../../images/test_set/fist/fist*.jpg")) 
path_o = sorted(glob.glob("../../images/test_set/open_hand/open_hand*.jpg"))   
path_h = sorted(glob.glob("../../images/test_set/horns/horns*.jpg"))   
path_t = sorted(glob.glob("../../images/test_set/thumbup/thumbup*.jpg"))

r1_count = 76
r2_count = 151
r3_count = 226
f_count = 301
fr1_count = 376
fr2_count = 451
fr3_count = 526

for (image_path_v, image_path_f, image_path_o, image_path_h, image_path_t) in zip(path_v, path_f, path_o, path_h, path_t):
    if r1_count > 150 or r2_count > 225 or r3_count > 300 or f_count > 375 or fr1_count > 450 or fr2_count > 525 or fr3_count > 600:
        break
    else:
        image_v = cv2.imread(image_path_v)
        image_vr1 = cv2.rotate(image_v, cv2.ROTATE_90_CLOCKWISE)
        image_vr2 = cv2.rotate(image_vr1, cv2.ROTATE_90_CLOCKWISE)
        image_vr3 = cv2.rotate(image_vr2, cv2.ROTATE_90_CLOCKWISE)
        image_vf = cv2.flip(image_v, 0)
        image_vfr1 = cv2.rotate(image_vf, cv2.ROTATE_90_CLOCKWISE)
        image_vfr2 = cv2.rotate(image_vfr1, cv2.ROTATE_90_CLOCKWISE)
        image_vfr3 = cv2.rotate(image_vfr2, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("../../images/test_set/victory/victory%d.jpg"%(r1_count), image_vr1)
        cv2.imwrite("../../images/test_set/victory/victory%d.jpg"%(r2_count), image_vr2)
        cv2.imwrite("../../images/test_set/victory/victory%d.jpg"%(r3_count), image_vr3)
        cv2.imwrite("../../images/test_set/victory/victory%d.jpg"%(f_count), image_vf)
        cv2.imwrite("../../images/test_set/victory/victory%d.jpg"%(fr1_count), image_vfr1)
        cv2.imwrite("../../images/test_set/victory/victory%d.jpg"%(fr2_count), image_vfr2)
        cv2.imwrite("../../images/test_set/victory/victory%d.jpg"%(fr3_count), image_vfr3)
        
        image_f = cv2.imread(image_path_f)
        image_fr1 = cv2.rotate(image_f, cv2.ROTATE_90_CLOCKWISE)
        image_fr2 = cv2.rotate(image_fr1, cv2.ROTATE_90_CLOCKWISE)
        image_fr3 = cv2.rotate(image_fr2, cv2.ROTATE_90_CLOCKWISE)
        image_ff = cv2.flip(image_f, 0)
        image_ffr1 = cv2.rotate(image_ff, cv2.ROTATE_90_CLOCKWISE)
        image_ffr2 = cv2.rotate(image_ffr1, cv2.ROTATE_90_CLOCKWISE)
        image_ffr3 = cv2.rotate(image_ffr2, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("../../images/test_set/fist/fist%d.jpg"%(r1_count), image_fr1)
        cv2.imwrite("../../images/test_set/fist/fist%d.jpg"%(r2_count), image_fr2)
        cv2.imwrite("../../images/test_set/fist/fist%d.jpg"%(r3_count), image_fr3)
        cv2.imwrite("../../images/test_set/fist/fist%d.jpg"%(f_count), image_ff)
        cv2.imwrite("../../images/test_set/fist/fist%d.jpg"%(fr1_count), image_ffr1)
        cv2.imwrite("../../images/test_set/fist/fist%d.jpg"%(fr2_count), image_ffr2)
        cv2.imwrite("../../images/test_set/fist/fist%d.jpg"%(fr3_count), image_ffr3)
       
       
        image_o = cv2.imread(image_path_o)
        image_or1 = cv2.rotate(image_o, cv2.ROTATE_90_CLOCKWISE)
        image_or2 = cv2.rotate(image_or1, cv2.ROTATE_90_CLOCKWISE)
        image_or3 = cv2.rotate(image_or2, cv2.ROTATE_90_CLOCKWISE)
        image_of = cv2.flip(image_o, 0)
        image_ofr1 = cv2.rotate(image_of, cv2.ROTATE_90_CLOCKWISE)
        image_ofr2 = cv2.rotate(image_ofr1, cv2.ROTATE_90_CLOCKWISE)
        image_ofr3 = cv2.rotate(image_ofr2, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("../../images/test_set/open_hand/open_hand%d.jpg"%(r1_count), image_or1)
        cv2.imwrite("../../images/test_set/open_hand/open_hand%d.jpg"%(r2_count), image_or2)
        cv2.imwrite("../../images/test_set/open_hand/open_hand%d.jpg"%(r3_count), image_or3)
        cv2.imwrite("../../images/test_set/open_hand/open_hand%d.jpg"%(f_count), image_of)
        cv2.imwrite("../../images/test_set/open_hand/open_hand%d.jpg"%(fr1_count), image_ofr1)
        cv2.imwrite("../../images/test_set/open_hand/open_hand%d.jpg"%(fr2_count), image_ofr2)
        cv2.imwrite("../../images/test_set/open_hand/open_hand%d.jpg"%(fr3_count), image_ofr3)
        
        image_h = cv2.imread(image_path_h)
        image_hr1 = cv2.rotate(image_h, cv2.ROTATE_90_CLOCKWISE)
        image_hr2 = cv2.rotate(image_hr1, cv2.ROTATE_90_CLOCKWISE)
        image_hr3 = cv2.rotate(image_hr2, cv2.ROTATE_90_CLOCKWISE)
        image_hf = cv2.flip(image_h, 0)
        image_hfr1 = cv2.rotate(image_hf, cv2.ROTATE_90_CLOCKWISE)
        image_hfr2 = cv2.rotate(image_hfr1, cv2.ROTATE_90_CLOCKWISE)
        image_hfr3 = cv2.rotate(image_hfr2, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("../../images/test_set/horns/horns%d.jpg"%(r1_count), image_hr1)
        cv2.imwrite("../../images/test_set/horns/horns%d.jpg"%(r2_count), image_hr2)
        cv2.imwrite("../../images/test_set/horns/horns%d.jpg"%(r3_count), image_hr3)
        cv2.imwrite("../../images/test_set/horns/horns%d.jpg"%(f_count), image_hf)
        cv2.imwrite("../../images/test_set/horns/horns%d.jpg"%(fr1_count), image_hfr1)
        cv2.imwrite("../../images/test_set/horns/horns%d.jpg"%(fr2_count), image_hfr2)
        cv2.imwrite("../../images/test_set/horns/horns%d.jpg"%(fr3_count), image_hfr3)
       
        image_t = cv2.imread(image_path_t)
        image_tr1 = cv2.rotate(image_t, cv2.ROTATE_90_CLOCKWISE)
        image_tr2 = cv2.rotate(image_tr1, cv2.ROTATE_90_CLOCKWISE)
        image_tr3 = cv2.rotate(image_tr2, cv2.ROTATE_90_CLOCKWISE)
        image_tf = cv2.flip(image_t, 0)
        image_tfr1 = cv2.rotate(image_tf, cv2.ROTATE_90_CLOCKWISE)
        image_tfr2 = cv2.rotate(image_tfr1, cv2.ROTATE_90_CLOCKWISE)
        image_tfr3 = cv2.rotate(image_tfr2, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("../../images/test_set/thumbup/thumbup%d.jpg"%(r1_count), image_tr1)
        cv2.imwrite("../../images/test_set/thumbup/thumbup%d.jpg"%(r2_count), image_tr2)
        cv2.imwrite("../../images/test_set/thumbup/thumbup%d.jpg"%(r3_count), image_tr3)
        cv2.imwrite("../../images/test_set/thumbup/thumbup%d.jpg"%(f_count), image_tf)
        cv2.imwrite("../../images/test_set/thumbup/thumbup%d.jpg"%(fr1_count), image_tfr1)
        cv2.imwrite("../../images/test_set/thumbup/thumbup%d.jpg"%(fr2_count), image_tfr2)
        cv2.imwrite("../../images/test_set/thumbup/thumbup%d.jpg"%(fr3_count), image_tfr3)
       
        r1_count += 1
        r2_count += 1
        r3_count += 1
        f_count += 1
        fr1_count += 1
        fr2_count += 1
        fr3_count += 1
    
    
    
    
