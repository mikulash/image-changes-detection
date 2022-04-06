import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

model = keras.models.load_model('./models/myModels/transferModelV2', custom_objects={'KerasLayer': hub.KerasLayer})
