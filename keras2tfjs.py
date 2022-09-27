import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

mobile_num = load_model('models/mobile_num.h5')

tfjs.converters.save_keras_model(mobile_num ,'models/tfjs_models')