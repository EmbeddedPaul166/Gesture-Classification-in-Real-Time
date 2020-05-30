import numpy as np
import cv2
import glob
import random
import os

# This script generates training, validation and test data

#Image dimensions
IMAGE_DIM = (150, 150, 1)

#Background threshold
B_TRESH = 30

#Translation values for downscaling
SCALE_ONE_RANGE = (125,150)
SCALE_TWO_RANGE = (100,125)
SCALE_THREE_RANGE = (75,100)
SCALE_FOUR_RANGE = (50,75)

#Image paths
PATH_OTR = sorted(glob.glob("training_set/open_hand/open_hand*.jpg"))   
PATH_CTR = sorted(glob.glob("training_set/closed_hand/closed_hand*.jpg")) 
PATH_OV = sorted(glob.glob("validation_set/open_hand/open_hand*.jpg"))   
PATH_CV = sorted(glob.glob("validation_set/closed_hand/closed_hand*.jpg")) 
PATH_OTE = sorted(glob.glob("test_set/open_hand/open_hand*.jpg"))   
PATH_CTE = sorted(glob.glob("test_set/closed_hand/closed_hand*.jpg")) 

def write_img(path, image, count):
    string = path + str(count) + ".jpg" 
    cv2.imwrite(string, image)

def add_noise(image, m_limit, s_limit):
    M = random.randint(0, m_limit) 
    S = random.randint(0, s_limit)
    noise = np.zeros(IMAGE_DIM)
    cv2.randn(noise, M, S)
    image = image + noise
    return image

def augment_data(path_o, path_c, mode, count):
    print("Augmenting", mode, "...") 
    count_o = count
    count_c = count
    image_background = cv2.imread("background/background.jpg")
    
    for (image_path_o, image_path_c) in zip(path_o, path_c):
        
        val = 0
        
        #Open hand images  
        images = []
        images_for_write = [] 
        
        #Normal
        image_o = cv2.imread(image_path_o)
        if mode == "training_set":
            image_o[image_o < B_TRESH] = val
        images.append(image_o)
        
        #Mirror-flip
        image_of = cv2.flip(image_o, 1)
        images_for_write.append(image_of)
        images.append(image_of)
        
        downscaled_image_list = []
        
        #Downscale
        if mode == "training_set":
            for image in images:
                for i in range(1,16):
                    scalex = random.randint(SCALE_ONE_RANGE[0], SCALE_ONE_RANGE[1])
                    scaley = random.randint(SCALE_ONE_RANGE[0], SCALE_ONE_RANGE[1])
                    deltax = random.randint(0, 150 - scalex)
                    deltay = random.randint(0, 150 - scaley)
                    im = image.copy()
                    image_b = image_background.copy()
                    im = cv2.resize(im, (scalex, scaley), cv2.INTER_LINEAR)
                    image_b[0+deltay:scaley+deltay,0+deltax:scalex+deltax] = im
                    downscaled_image_list.append(image_b)
                for i in range(1,16):
                    scalex = random.randint(SCALE_TWO_RANGE[0], SCALE_TWO_RANGE[1])
                    scaley = random.randint(SCALE_TWO_RANGE[0], SCALE_TWO_RANGE[1])
                    deltax = random.randint(0, 150 - scalex)
                    deltay = random.randint(0, 150 - scaley)
                    im = image.copy()
                    image_b = image_background.copy()
                    im = cv2.resize(im, (scalex, scaley), cv2.INTER_LINEAR)
                    image_b[0+deltay:scaley+deltay,0+deltax:scalex+deltax] = im
                    downscaled_image_list.append(image_b)
                for i in range(1,16):
                    scalex = random.randint(SCALE_THREE_RANGE[0], SCALE_THREE_RANGE[1])
                    scaley = random.randint(SCALE_THREE_RANGE[0], SCALE_THREE_RANGE[1])
                    deltax = random.randint(0, 150 - scalex)
                    deltay = random.randint(0, 150 - scaley)
                    im = image.copy()
                    image_b = image_background.copy()
                    im = cv2.resize(im, (scalex, scaley), cv2.INTER_LINEAR)
                    image_b[0+deltay:scaley+deltay,0+deltax:scalex+deltax] = im
                    downscaled_image_list.append(image_b)
                for i in range(1,16):
                    scalex = random.randint(SCALE_FOUR_RANGE[0], SCALE_FOUR_RANGE[1])
                    scaley = random.randint(SCALE_FOUR_RANGE[0], SCALE_FOUR_RANGE[1])
                    deltax = random.randint(0, 150 - scalex)
                    deltay = random.randint(0, 150 - scaley)
                    im = image.copy()
                    image_b = image_background.copy()
                    im = cv2.resize(im, (scalex, scaley), cv2.INTER_LINEAR)
                    image_b[0+deltay:scaley+deltay,0+deltax:scalex+deltax] = im
                    downscaled_image_list.append(image_b)
            
            images.extend(downscaled_image_list)
            images_for_write.extend(downscaled_image_list)
            
            #Change brightness
            for i, image in enumerate(images_for_write):
                #Darken
                brightness_delta = random.randint(0, 90)
                part = image[image > 0]
                part[part < brightness_delta] = np.amin(part)
                part[part >= brightness_delta] -= brightness_delta 
                images_for_write[i][images_for_write[i] > 0] = part
                
            val = random.randint(0,255)
            image_o[image_o == 0] = val
            im = add_noise(image_o, 20, 30)
            cv2.imwrite(image_path_o, im)
        
        #Write images
        for image in images_for_write:
            im = image.astype("uint8")
            if mode == "training_set":
                val = random.randint(0,255)
                im[im == 0] = val
                im = add_noise(im, 20, 30)
            write_img(mode + "/open_hand/open_hand", im, count_o)
            count_o += 1
        
        val = 0
        
        #Closed hand images
        images = []
        images_for_write = [] 
        
        #Normal
        image_c = cv2.imread(image_path_c)
        if mode == "training_set":
            image_c[image_c < B_TRESH] = val
        images.append(image_c)
        
        #Mirror-flip
        image_cf = cv2.flip(image_c, 1)
        images_for_write.append(image_cf)
        images.append(image_cf)
        
        downscaled_image_list = []
        
        #Downscale
        if mode == "training_set":
            for image in images:
                for i in range(1,11):
                    scalex = random.randint(SCALE_ONE_RANGE[0], SCALE_ONE_RANGE[1])
                    scaley = random.randint(SCALE_ONE_RANGE[0], SCALE_ONE_RANGE[1])
                    deltax = random.randint(0, 150 - scalex)
                    deltay = random.randint(0, 150 - scaley)
                    im = image.copy()
                    image_b = image_background.copy()
                    im = cv2.resize(im, (scalex, scaley), cv2.INTER_LINEAR)
                    image_b[0+deltay:scaley+deltay,0+deltax:scalex+deltax] = im
                    downscaled_image_list.append(image_b)
                for i in range(1,11):
                    scalex = random.randint(SCALE_TWO_RANGE[0], SCALE_TWO_RANGE[1])
                    scaley = random.randint(SCALE_TWO_RANGE[0], SCALE_TWO_RANGE[1])
                    deltax = random.randint(0, 150 - scalex)
                    deltay = random.randint(0, 150 - scaley)
                    im = image.copy()
                    image_b = image_background.copy()
                    im = cv2.resize(im, (scalex, scaley), cv2.INTER_LINEAR)
                    image_b[0+deltay:scaley+deltay,0+deltax:scalex+deltax] = im
                    downscaled_image_list.append(image_b)
                for i in range(1,11):
                    scalex = random.randint(SCALE_THREE_RANGE[0], SCALE_THREE_RANGE[1])
                    scaley = random.randint(SCALE_THREE_RANGE[0], SCALE_THREE_RANGE[1])
                    deltax = random.randint(0, 150 - scalex)
                    deltay = random.randint(0, 150 - scaley)
                    im = image.copy()
                    image_b = image_background.copy()
                    im = cv2.resize(im, (scalex, scaley), cv2.INTER_LINEAR)
                    image_b[0+deltay:scaley+deltay,0+deltax:scalex+deltax] = im
                    downscaled_image_list.append(image_b)
                for i in range(1,11):
                    scalex = random.randint(SCALE_FOUR_RANGE[0], SCALE_FOUR_RANGE[1])
                    scaley = random.randint(SCALE_FOUR_RANGE[0], SCALE_FOUR_RANGE[1])
                    deltax = random.randint(0, 150 - scalex)
                    deltay = random.randint(0, 150 - scaley)
                    im = image.copy()
                    image_b = image_background.copy()
                    im = cv2.resize(im, (scalex, scaley), cv2.INTER_LINEAR)
                    image_b[0+deltay:scaley+deltay,0+deltax:scalex+deltax] = im
                    downscaled_image_list.append(image_b)
            
            images.extend(downscaled_image_list)
            images_for_write.extend(downscaled_image_list)
            
            #Change brightness
            for i, image in enumerate(images_for_write):
                #Darken
                brightness_delta = random.randint(0, 90)
                part = image[image > 0]
                part[part < brightness_delta] = np.amin(part)
                part[part >= brightness_delta] -= brightness_delta 
                images_for_write[i][images_for_write[i] > 0] = part
                
            val = random.randint(0,255)
            image_c[image_c == 0] = val
            im = add_noise(image_c, 20, 30)
            cv2.imwrite(image_path_c, im)
        
        #Write images
        for image in images_for_write:
            im = image.astype("uint8")
            if mode == "training_set":
                val = random.randint(0,255)
                im[im == 0] = val
                im = add_noise(im, 20, 30)
            write_img(mode + "/closed_hand/closed_hand", im, count_c)
            count_c += 1

def augment_empty_background_class(count):
    os.mkdir("training_set/none/")
    black = np.zeros((IMAGE_DIM))
    for i in range(count, 10001):
        im = black.copy()
        val = random.randint(0,255)
        im[im == 0] = val
        im = add_noise(im, 127, 127)
        write_img("training_set/none/none", im, count) 
        count += 1
        
def augment_wrong_gesture_class(count):
    count_drop = 1
    count = 1
    count_ro = 1
    count_rc = 1 
    os.mkdir("training_set/wrong_gesture/")  
    path_otr = sorted(glob.glob("training_set/open_hand/open_hand*.jpg"))   
    path_ctr = sorted(glob.glob("training_set/closed_hand/closed_hand*.jpg")) 
    for (image_path_o, image_path_c) in zip(path_otr, path_ctr):
        if count_drop == 1:
            if count % 2 == 0:
                image_o = cv2.imread(image_path_o)
                if count_ro == 1:
                    im = cv2.rotate(image_o, cv2.ROTATE_90_CLOCKWISE)
                    count_ro += 1
                elif count_ro == 2:
                    im = cv2.rotate(image_o, cv2.ROTATE_90_COUNTERCLOCKWISE) 
                    count_ro += 1
                else:
                    im = cv2.rotate(image_o, cv2.ROTATE_180)
                    count_ro = 1     
            else:
                image_c = cv2.imread(image_path_c)
                if count_rc == 1:
                    im = cv2.rotate(image_c, cv2.ROTATE_90_CLOCKWISE)
                    count_rc += 1
                elif count_rc == 2:
                    im = cv2.rotate(image_c, cv2.ROTATE_90_COUNTERCLOCKWISE) 
                    count_rc += 1
                else:
                    im = cv2.rotate(image_c, cv2.ROTATE_180)
                    count_rc = 1
           
            write_img("training_set/wrong_gesture/wrong_gesture", im, count) 
            count_drop += 1
            count += 1
        else:
            count_drop = 1

augment_data(PATH_OTR, PATH_CTR, "training_set", 601)
augment_data(PATH_OV, PATH_CV, "validation_set", 1601)
augment_data(PATH_OTE, PATH_CTE, "test_set", 1201)
augment_empty_background_class(1)
#augment_wrong_gesture_class(1)


print("Done")
