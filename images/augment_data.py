import numpy as np
import cv2
import glob
import random

# Procedure for data augmentation of training data:
#
# 1. Load normal image
# 2. Mirror-flip it
# 3. Downscale it multiple times and paste it onto the blank background with multiple translations per size
# 4. Add random backgrounds to loaded image and generated images
# 5. Write all the images
# 6. Repeat for another image
#
# Procedure for test and validation data:
#
# 1. Load normal image
# 2. Mirror-flip it
# 3. Write all generated images
# 4. Repeat for another image
#
#Also for "none" training dir, there are rotated gestures, so that the net will only recognize one that are vertical

#Image dimensions
IMAGE_DIM = (150, 150, 1)

#Noise parameters
M = 2
S = 4

#Gaussian blur kernel
G_KERNEL = (3,3)

#Background threshold
B_TRESH = 30

BRI_MIN_LIM = 40
BRI_MAX_LIM = 120

#Translation values for downscaling
SHIFTS_ONE = [0, 25]
SHIFTS_TWO = [0, 25, 50]
SHIFTS_THREE = [0, 25, 50, 75]

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

def add_noise(image):
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
                for delta_x in SHIFTS_ONE:
                    for delta_y in SHIFTS_ONE:
                        im = image.copy()
                        image_b = image_background.copy()
                        im = cv2.resize(im, (125, 125), cv2.INTER_LINEAR)
                        image_b[0+delta_y:125+delta_y,0+delta_x:125+delta_x] = im
                        downscaled_image_list.append(image_b)
                for delta_x in SHIFTS_TWO:
                    for delta_y in SHIFTS_TWO:
                        im = image.copy()
                        image_b = image_background.copy()
                        im = cv2.resize(im, (100, 100), cv2.INTER_LINEAR)
                        image_b[0+delta_y:100+delta_y,0+delta_x:100+delta_x] = im
                        downscaled_image_list.append(image_b)
                for delta_x in SHIFTS_THREE:
                    for delta_y in SHIFTS_THREE:
                        im = image.copy()
                        image_b = image_background.copy()
                        im = cv2.resize(im, (75, 75), cv2.INTER_LINEAR)
                        image_b[0+delta_y:75+delta_y,0+delta_x:75+delta_x] = im
                        downscaled_image_list.append(image_b)
            
            images.extend(downscaled_image_list)
            images_for_write.extend(downscaled_image_list)
            
            #Change brightness
            brightness_changed_image_list = []
            
            for image in images:
                #Brighten
                im = image.copy()                
                brightness_delta = 30
                part = im[im > 0]
                part[part + brightness_delta > 255] = np.amax(part)
                part[part + brightness_delta < 255] += brightness_delta 
                im[im > 0] = part
                brightness_changed_image_list.append(im)
                
                #Darken
                im = image.copy()                
                brightness_delta = 30
                part = im[im > 0]
                part[part < brightness_delta] = np.amin(part)
                part[part >= brightness_delta] -= brightness_delta 
                im[im > 0] = part
                brightness_changed_image_list.append(im)
                
                #Darken
                im = image.copy()                
                brightness_delta = 60
                part = im[im > 0]
                part[part < brightness_delta] = np.amin(part)
                part[part >= brightness_delta] -= brightness_delta 
                im[im > 0] = part
                brightness_changed_image_list.append(im)
                
                #Darken
                im = image.copy()                
                brightness_delta = 90
                part = im[im > 0]
                part[part < brightness_delta] = np.amin(part)
                part[part >= brightness_delta] -= brightness_delta 
                im[im > 0] = part
                brightness_changed_image_list.append(im)
                
            images.extend(brightness_changed_image_list)
            images_for_write.extend(brightness_changed_image_list)
            
            val = random.randint(0,255)
            image_o[image_o == 0] = val
            im = add_noise(image_o)
            cv2.imwrite(image_path_o, im)
        
        #Write images
        for image in images_for_write:
            im = image.astype("uint8")
            if mode == "training_set":
                val = random.randint(0,255)
                im[im == 0] = val
                im = add_noise(im)
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
                for delta_x in SHIFTS_ONE:
                    for delta_y in SHIFTS_ONE:
                        im = image.copy()
                        image_b = image_background.copy()
                        im = cv2.resize(im, (125, 125), cv2.INTER_LINEAR)
                        image_b[0+delta_y:125+delta_y,0+delta_x:125+delta_x] = im
                        downscaled_image_list.append(image_b)
                for delta_x in SHIFTS_TWO:
                    for delta_y in SHIFTS_TWO:
                        im = image.copy()
                        image_b = image_background.copy()
                        im = cv2.resize(im, (100, 100), cv2.INTER_LINEAR)
                        image_b[0+delta_y:100+delta_y,0+delta_x:100+delta_x] = im
                        downscaled_image_list.append(image_b)
                for delta_x in SHIFTS_THREE:
                    for delta_y in SHIFTS_THREE:
                        im = image.copy()
                        image_b = image_background.copy()
                        im = cv2.resize(im, (75, 75), cv2.INTER_LINEAR)
                        image_b[0+delta_y:75+delta_y,0+delta_x:75+delta_x] = im
                        downscaled_image_list.append(image_b)
            
            images.extend(downscaled_image_list)
            images_for_write.extend(downscaled_image_list)
            
            #Change brightness
            brightness_changed_image_list = []
            
            for image in images:
                #Brighten
                im = image.copy()                
                brightness_delta = 30
                part = im[im > 0]
                part[part + brightness_delta > 255] = np.amax(part)
                part[part + brightness_delta < 255] += brightness_delta 
                im[im > 0] = part
                brightness_changed_image_list.append(im)
                
                #Darken
                im = image.copy()                
                brightness_delta = 30
                part = im[im > 0]
                part[part < brightness_delta] = np.amin(part)
                part[part >= brightness_delta] -= brightness_delta 
                im[im > 0] = part
                brightness_changed_image_list.append(im)
                
                #Darken
                im = image.copy()                
                brightness_delta = 60
                part = im[im > 0]
                part[part < brightness_delta] = np.amin(part)
                part[part >= brightness_delta] -= brightness_delta 
                im[im > 0] = part
                brightness_changed_image_list.append(im)
                
                #Darken
                im = image.copy()                
                brightness_delta = 90
                part = im[im > 0]
                part[part < brightness_delta] = np.amin(part)
                part[part >= brightness_delta] -= brightness_delta 
                im[im > 0] = part
                brightness_changed_image_list.append(im)
                
            images.extend(brightness_changed_image_list)
            images_for_write.extend(brightness_changed_image_list)
            
            val = random.randint(0,255)
            image_c[image_c == 0] = val
            im = add_noise(image_c)
            cv2.imwrite(image_path_c, im)
        
        #Write images
        for image in images_for_write:
            im = image.astype("uint8")
            if mode == "training_set":
                val = random.randint(0,255)
                im[im == 0] = val
                im = add_noise(im)
            write_img(mode + "/closed_hand/closed_hand", im, count_c)
            count_c += 1

augment_data(PATH_OTR, PATH_CTR, "training_set", 301)
augment_data(PATH_OV, PATH_CV, "validation_set", 1201)
augment_data(PATH_OTE, PATH_CTE, "test_set", 801)


PATH_OTR = sorted(glob.glob("training_set/open_hand/open_hand*.jpg"))   
PATH_CTR = sorted(glob.glob("training_set/closed_hand/closed_hand*.jpg")) 

count_b = 1
count_ro = 1
count_rc = 1
for (image_path_o, image_path_c) in zip(PATH_OTR, PATH_CTR):
    if count_b % 2 == 0:
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
        
    write_img("training_set/none/none", im, count_b)
    count_b += 1
    
print("Done")
