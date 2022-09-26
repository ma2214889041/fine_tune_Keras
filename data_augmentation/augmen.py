import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from test_img import plot_images

gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1,
    channel_shift_range=10., horizontal_flip=True)

# random choice
chosen_image = random.choice(os.listdir('../data/cat_dog/train/dog'))
image_path = '../data/cat_dog/train/dog/' + chosen_image

#original image looks like
image = np.expand_dims(plt.imread(image_path),0)
plt.imshow(image[0])

# generate batches of augmented image
aug_iter = gen.flow(image)
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
# plot the augmented images
plot_images(aug_images)

# Save Augmented Data
aug_iter = gen.flow(image, save_to_dir='data/cat_dog/train/dog', save_prefix='aug-image-', save_format='jpeg')

