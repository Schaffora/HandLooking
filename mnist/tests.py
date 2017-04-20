from keras.models import model_from_json
import keras
import numpy as np
import cv2

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

img_width, img_height = 28, 28
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
verbose = 0

img =cv2.imread("3.png")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image = gray_image.reshape((1,)+ gray_image.shape+(1,))
#print('prediction of [1, 1]: ', loaded_model.predict_classes(img, verbose=verbose))
y_proba = loaded_model.predict(gray_image)
print(y_proba)
y_classes = loaded_model.predict_classes(gray_image)
print(y_classes)
#y_classes = keras.np_utils.probas_to_classes(y_proba)
