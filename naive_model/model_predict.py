import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix


test_path = '../data/cat_dog/test'
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path,
    target_size=(224, 224),
    batch_size=10,
    classes=['cat','dog']
)

'''
# testing imgs
from test_img import plot_images
test_imgs, test_labels = next(test_batches)
plot_images(test_imgs)
print(test_labels)
'''

#Read model
model = load_model('models/cat_dog.h5')

#Prediction
predictions =model.predict(x=test_batches, verbose=0)
np.round(predictions) # to 1 or 0
cm=confusion_matrix(y_true=test_batches.classes,y_pred=np.argmax(predictions, axis=-1))

print(cm)