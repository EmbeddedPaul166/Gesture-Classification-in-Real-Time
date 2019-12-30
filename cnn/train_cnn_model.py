from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

model = None
model_history = None
input_shape = (45, 45, 1)
is_verbose = 1

batch_size = 128
number_of_epochs = 30

def load_dataset(path_to_directory):
    datagen = ImageDataGenerator()
    dataset = datagen.flow_from_directory(path_to_directory, input_shape[:2], batch_size = batch_size,
                                          color_mode = "grayscale", class_mode = "categorical", shuffle = True)
    return dataset

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = "relu", padding = "valid", input_shape = input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation = "relu", padding = "valid"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation = "relu", padding = "valid"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation = "softmax"))
    
    model.compile(optimizer = Adam(), loss = categorical_crossentropy, metrics = ["accuracy"])
    return model

def train_and_evaluate_model():
    training_dataset = load_dataset("../images/training_set/")
    validation_dataset = load_dataset("../images/validation_set/")
    test_dataset = load_dataset("../images/test_set/")
    
    model = define_model()
    model_history = model.fit_generator(training_dataset, steps_per_epoch = len(training_dataset),
                                        epochs = number_of_epochs, verbose = is_verbose,
                                        validation_data = validation_dataset, shuffle = True)
    score = model.evaluate_generator(test_dataset, verbose = 1)
    print("\nTest loss:", score[0])
    print("\nTest accuracy:", score[1])
    model.save('cnn_model.h5')

train_and_evaluate_model()

