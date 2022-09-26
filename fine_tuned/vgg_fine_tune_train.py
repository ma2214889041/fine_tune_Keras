import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# vgg
vgg16_model = tf.keras.applications.vgg16.VGG16()
# vgg16_model.summary()

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


# add vgg layer on my model without output layer
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
#model.summary()

# freeze layer on my model, add output layer
for layer in model.layers:
    layer.trainable = False
model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=train_batches,
          validation_data=valid_batches,
          epochs=5,
          verbose=2)

# Save model
if os.path.isfile('../models/vgg_cat_dog.h5') is False:
    model.save('../models/vgg_cat_dog.h5')