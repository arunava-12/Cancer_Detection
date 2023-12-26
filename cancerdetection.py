import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import math

warnings.filterwarnings('ignore')

ROOT_DIR = r"C:\Users\HP\Documents\cancer projects\data"
number_of_images = {}

for dir in os.listdir(ROOT_DIR):
    number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))

print(number_of_images.items())

def dataFolder(p, split):
    train_path = os.path.join(".", p)
    
    if not os.path.exists(train_path):
        os.mkdir(train_path)

        for dir in os.listdir(ROOT_DIR):
            dir_path = os.path.join(train_path, dir)
            os.mkdir(dir_path)

            images = os.listdir(os.path.join(ROOT_DIR, dir))
            selected_images = np.random.choice(images, size=math.floor(split * 0.7 * len(images)), replace=False)

            for img in selected_images:
                source_path = os.path.join(ROOT_DIR, dir, img)
                destination_path = os.path.join(train_path, dir, img)
                shutil.move(source_path, destination_path)
    else:
        print(f"{p} Folder exists")


# Example call with p as "train", "val", "test" and split as 0.7, 0.15, 0.15
dataFolder("train", 0.7)
dataFolder("val", 0.15)
dataFolder("test", 0.15)

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras

model = Sequential()

model.add(Conv2D(filters=16, kernel_size = (3,3), activation= 'relu', input_shape = (224,224,3)))

model.add(Conv2D(filters=36, kernel_size = (3,3), activation= 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters=64, kernel_size = (3,3), activation= 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters=128, kernel_size = (3,3), activation= 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(rate = 0.25))

model.add(Flatten())
model.add(Dense(units = 64, activation = 'relu'))

model.add(Dropout(rate = 0.25))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.summary()

model.compile(optimizer = 'adam', loss = keras.losses.binary_crossentropy, metrics = ['accuracy'])

def preprocessingImages1(path):
    
    image_data = ImageDataGenerator(zoom_range= 0.2, shear_range = 0.2,rescale = 1/255, horizontal_flip = True)
    image = image_data.flow_from_directory(directory = path, target_size = (224,224), batch_size = 32, class_mode = 'binary')
    
    return image

path = (r"C:\Users\HP\Documents\cancer projects\train")
train_data = preprocessingImages1(path)

def preprocessingImages2(path):
    
    image_data = ImageDataGenerator(rescale = 1/255)
    image = image_data.flow_from_directory(directory = path, target_size = (224,224), batch_size = 32, class_mode = 'binary')
    
    return image

path = (r"C:\Users\HP\Documents\cancer projects\test")
test_data = preprocessingImages2(path)

def preprocessingImages3(path):
    
    image_data = ImageDataGenerator(rescale = 1/255)
    image = image_data.flow_from_directory(directory = path, target_size = (224,224), batch_size = 32, class_mode = 'binary')
    
    return image

path = (r"C:\Users\HP\Documents\cancer projects\val")
val_data = preprocessingImages3(path)

from keras.callbacks import ModelCheckpoint, EarlyStopping
es = EarlyStopping(monitor = "val_accuracy", min_delta = 0.01, patience = 3, verbose = 1, mode = 'auto')

mc = ModelCheckpoint(monitor = "val_accuracy", filepath = "bestmodel.h5",verbose = 1, save_best_only = True, mode = "auto")

cd = [es,mc]

hs = model.fit_generator(generator = train_data, steps_per_epoch = 8, epochs = 30, verbose = 1, validation_data = val_data, validation_steps = 16, callbacks = cd)

h = hs.history
print(h.keys())

loss = hs.history['loss']
accuracy = hs.history['accuracy']
val_loss = hs.history['val_loss']
val_accuracy = hs.history['val_accuracy']
print(loss)
print(accuracy)
print(val_loss)
print(val_accuracy)

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'])
plt.title('acc vs val_acc')
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'])
plt.title('loss vs val_loss')
plt.show()

from keras.models import load_model
model = load_model(r"C:\Users\HP\Documents\cancer projects\bestmodel.h5")
acc = model.evaluate_generator(test_data)[1]
print(f"Accuracy is: {acc * 100}%")


