from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import tensorflow as tf
import numpy as np
import os

#Training parameters
INPUT_SHAPE = (150, 150, 1)
IMAGE_SIZE = (150, 150)
IS_VERBOSE = 1
BATCH_SIZE = 16
NUMBER_OF_EPOCHS = 1000000
STEPS_PER_EPOCH = 1000
VALIDATION_STEPS = 150
TEST_STEPS = 100
CLASS_NAMES = np.array(["closed_hand", "open_hand"])

AUTOTUNE=tf.data.experimental.AUTOTUNE

CALLBACK_LIST = [EarlyStopping(monitor = "val_accuracy", patience = 20, restore_best_weights = True)] 

def get_label(file_path):
    parts = tf.strings.split(file_path, "/")
    return parts[-2] == CLASS_NAMES

def decode_image(img):
    img = tf.io.decode_jpeg(img, channels = 1, dct_method='INTEGER_ACCURATE')
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_image(img)
    return img, label

def prepare_for_training(ds, cache = True, shuffle_buffer_size = 1000): 
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
    else:
        ds = ds.cache()
        
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

#Dataset loading
def create_dataset(path):
    list_dataset = tf.data.Dataset.list_files(path)
    dataset = list_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    dataset = prepare_for_training(dataset)
    return dataset

#Model creation
def create_model():
    model = Sequential()
     
    model.add(Conv2D(48, (3, 3), activation = "relu", padding = "valid", input_shape = INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Conv2D(48, (3, 3), activation = "relu", padding = "valid"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2, 2), strides = 2))
    
    model.add(Conv2D(96, (3, 3), activation = "relu", padding = "valid"))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (3, 3), activation = "relu", padding = "valid"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2, 2), strides = 2))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation = "relu"))
    model.add(BatchNormalization())
    
    model.add(Dense(2, activation = "sigmoid"))
    
    model.compile(optimizer = SGD(), loss = "binary_crossentropy", metrics = ["accuracy"])
    
    return model

#Model training and evaluation
def train_and_evaluate_model():
    training_dataset = create_dataset("../images/training_set/*/*")
    validation_dataset = create_dataset("../images/validation_set/*/*")
    test_dataset = create_dataset("../images/test_set/*/*")
    
    model = create_model()
     
    model.fit(training_dataset, epochs = NUMBER_OF_EPOCHS, steps_per_epoch = STEPS_PER_EPOCH, validation_data = validation_dataset, validation_steps = VALIDATION_STEPS, shuffle = True, callbacks = CALLBACK_LIST, use_multiprocessing = True, verbose = IS_VERBOSE)
    
    score = model.evaluate(test_dataset, steps = TEST_STEPS, verbose = IS_VERBOSE)
    
    print("\nTest loss:", score[0])
    print("\nTest accuracy:", score[1])
    
    model.save("cnn_model.h5")
    
train_and_evaluate_model()
