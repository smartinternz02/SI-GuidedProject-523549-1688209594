# Importing required libs
from keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# Loading model
model = load_model("C:/Users/user/stage1.h5")


# Preparing and pre-processing the image
def preprocess_img(img_path):
    image_ = load_img(img_path, target_size=(224, 224))
    image_ = img_to_array(image_)
    image_ = preprocess_input(image_)
    data = []
    data.append(image_)
    data = np.array(data, dtype="float32")
    return data


# Predicting function
def predict_result(predict):
    pred = model.predict(predict)
    if pred[0][0]>pred[0][1]:
        return "damaged"
    else:
        return "not damaged"