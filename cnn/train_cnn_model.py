from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
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
BATCH_SIZE = 32
NUMBER_OF_EPOCHS = 10000
STEPS_PER_EPOCH = 5200
VALIDATION_STEPS = 200
TEST_STEPS = 150
CLASS_NAMES = np.array(["closed_hand", "open_hand", "none"])

AUTOTUNE=tf.data.experimental.AUTOTUNE

CALLBACK_LIST = [ModelCheckpoint(filepath = "checkpoints/cnn_model{epoch}.h5", monitor="val_acc"),
                 TensorBoard(log_dir = "./tensorboard_logs", histogram_freq = 1)]

#EarlyStopping(monitor = "val_accuracy", min_delta = 0.01, patience = 15, restore_best_weights = True)

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

def prepare_for_training(ds, cache = True, shuffle_buffer_size = 8000): 
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
     
    model.add(Conv2D(48, (3, 3), activation = "relu", padding = "same", input_shape = INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Conv2D(48, (3, 3), activation = "relu", padding = "same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation = "relu", padding = "same"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation = "relu", padding = "same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2, 2)))
    
    model.add(Conv2D(96, (3, 3), activation = "relu", padding = "same"))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (3, 3), activation = "relu", padding = "same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(3, activation = "softmax"))
    
    model.compile(optimizer = Adam(learning_rate = 0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    return model

#Model training and evaluation
def train_and_evaluate_model():
    training_dataset = create_dataset("../images/training_set/*/*")
    validation_dataset = create_dataset("../images/validation_set/*/*")
    test_dataset = create_dataset("../images/test_set/*/*")
    
    model = create_model()
    
    model.fit(training_dataset, epochs = NUMBER_OF_EPOCHS, steps_per_epoch = STEPS_PER_EPOCH, validation_data = validation_dataset, validation_steps = VALIDATION_STEPS, shuffle = True, callbacks = CALLBACK_LIST, use_multiprocessing = True, verbose = IS_VERBOSE)
    
    #score = model.evaluate(test_dataset, steps = TEST_STEPS, verbose = IS_VERBOSE)
    
    #print("\nTest loss:", score[0])
    #print("\nTest accuracy:", score[1])
    
    #model.save("cnn_model.h5")
    
    #with open('test_results', 'a') as out:
    #    out.write("Test loss: " + str(score[0]))
    #    out.write("Test accuracy: " + str(score[1]))
    
train_and_evaluate_model()
