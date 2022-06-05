# Correct Code
from keras.preprocessing.image import img_to_array
from flask import Flask, render_template, request
import tensorflow
from keras.models import load_model
import numpy as np
import os
from keras.preprocessing import image
import cv2
#from untitled1 import x_train

app = Flask(__name__)

model = load_model('sign_language.h5')

dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
       10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
       20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

model.make_predict_function()



@app.route("/")
def main():
    return render_template("background.html")

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

@app.route("/continue")
def new():
    return render_template("index.html")

def classify(image):
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)
    idx = np.argmax(proba)
    return alphabet[idx]





@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        print(img_path)
        image = cv2.imread(img_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p = classify(img)
        return render_template("index.html", prediction=p)

if __name__ == '__main__':
    app.run()