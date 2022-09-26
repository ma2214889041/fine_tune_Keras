import os
import shutil
import random
import glob
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

os.chdir('../data/cat_dog')

if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')

    for i in random.sample(glob.glob('cat*'),1000):
        shutil.move(i,'train/cat')
    for i in random.sample(glob.glob('dog*'),1000):
        shutil.move(i,'train/dog')
    for i in random.sample(glob.glob('cat*'),200):
        shutil.move(i,'valid/cat')
    for i in random.sample(glob.glob('dog*'),200):
        shutil.move(i,'valid/dog')
    for i in random.sample(glob.glob('cat*'),100):
        shutil.move(i,'test/cat')
    for i in random.sample(glob.glob('dog*'),100):
        shutil.move(i,'test/dog')