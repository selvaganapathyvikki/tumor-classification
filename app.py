
from flask import Flask, render_template, request
# import requests
import werkzeug
import keras
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
# from keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.optimizers import Adam # - Works
# import tensorflow as tf
import cv2
import os
import numpy as np

app = Flask(__name__)


loaded_model_classification = keras.models.load_model("final_model_1")


loaded_model_valid_checker = keras.models.load_model("final_model_valid_mri_1")
loaded_model_classification.summary()

@app.route('/')
def home() :
    return "welcome to tumor-classification-app"
@app.route('/predict', methods=["POST"])
def index():
    # print("hii")
    image = request.files['image']
    filename = werkzeug.utils.secure_filename(image.filename)
    image.save("./uploaded_images/"+filename)
    # npimg = np.fromstring(filen, np.uint8)
    # img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = cv2.imread("./uploaded_images/"+filename)[...,::-1] 
    
    img_size = 224

    resized_arr_img = cv2.resize(img, (img_size, img_size))
    resized_arr = np.array(resized_arr_img) / 255

    resized_arr = resized_arr.reshape(-1, img_size, img_size, 3)
    predictions = loaded_model_classification.predict(resized_arr)
    # print(predictions)
    predictions = np.argmax(predictions, axis=1)
    predictions = predictions.reshape(1,-1)[0]

    # print(predictions)

    if 0 in predictions :
        return ("Yes")
    else :
        return ("No")


@app.route('/valid', methods=["POST"])
def new():

    image = request.files['image']
    filename = werkzeug.utils.secure_filename(image.filename)
    image.save("./uploaded_images/"+filename)
    # npimg = np.fromstring(filen, np.uint8)
    # img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = cv2.imread("./uploaded_images_1/"+filename)[...,::-1] 
    
    img_size = 224

    resized_arr_img = cv2.resize(img, (img_size, img_size))
    resized_arr = np.array(resized_arr_img) / 255

    resized_arr = resized_arr.reshape(-1, img_size, img_size, 3)
    predictions = loaded_model_valid_checker.predict(resized_arr)
    # print(predictions)
    predictions = np.argmax(predictions, axis=1)
    predictions = predictions.reshape(1,-1)[0]

    # print(predictions)

    if 0 in predictions :
        return ("Valid MRI Image")
    else :
        return ("Invalid MRI Image")


if __name__ == "__main__":
    app.run(debug=True)
