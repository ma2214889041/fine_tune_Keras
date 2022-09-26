import os.path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# % matplotlib inline

train_path = '../data/cat_dog/train'
valid_path = '../data/cat_dog/valid'
test_path = '../data/cat_dog/test'

# keras generator
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path,
    target_size=(224, 224),
    batch_size=10,
    classes=['cat','dog']
)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=valid_path,
    target_size=(224, 224),
    batch_size=10,
    classes=['cat','dog']
)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path,
    target_size=(224, 224),
    batch_size=10,
    classes=['cat','dog']
)

'''
# testing imgs
imgs, labels = next(train_batches)
plot_images(imgs)
print(labels)
'''

# Model creation

model = Sequential([
    Conv2D(filters=32,
           kernel_size=(3,3),
           activation='relu',
           padding='same',
           input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2,2),strides=2),
    Conv2D(filters=64,
           kernel_size=(3,3),
           activation='relu',
           padding='same'),
    MaxPool2D(pool_size=(2,2),strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

# print model struct
print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=train_batches,
          validation_data=valid_batches,
          epochs=5,
          verbose=2)


# Save model
if os.path.isfile('../models/cat_dog.h5') is False:
    model.save('models/cat_dog.h5')



