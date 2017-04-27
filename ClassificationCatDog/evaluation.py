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

img_width, img_height = 150, 150
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
verbose = 0

imLoad = []

imLoad.append([cv2.imread("chat.jpg"), "chat.jpg"])
imLoad.append([cv2.imread("dog.807.jpg"), "dog.807.jpg"])
imLoad.append([cv2.imread("cat.876.jpg"), "cat.876.jpg"])
imLoad.append([cv2.imread("cat.58.jpg"), "cat.58.jpg"])
imLoad.append([cv2.imread("cat.1.jpg"), "cat.1.jpg"])
imLoad.append([cv2.imread("cat2.jpg"), "cat2.jpg"])
imLoad.append([cv2.imread("cat.2294.jpg"), "cat.2294.jpg"])


numCat=0
numDog=0
for img, name in imLoad:
    img = cv2.resize(img,( 150, 150))
    print("===", name, "===")
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_image = gray_image.reshape((1,) + gray_image.shape + (1,))
    gray_image = img.reshape((1,) + img.shape)
    # print('prediction of [1, 1]: ', loaded_model.predict_classes(img, verbose=verbose))


    # seem to be: 0 cats, 1 dogs
    y_proba = loaded_model.predict(gray_image)
    print(y_proba)
    y_classes = loaded_model.predict_classes(gray_image)
    print(y_classes)
    if y_classes==0:
        numCat+=1
    else:
        numDog+=1
    # y_classes = keras.np_utils.probas_to_classes(y_proba)

print("cat expected:",6)
print("cat found:",numCat)
print("dog expected:",1)
print("dog found:",numDog)