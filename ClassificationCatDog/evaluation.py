from keras.models import model_from_json
import keras
import numpy as np
import cv2
import os
import sys


# Eval images find in /eval/ directories.
def multipleEval():
    # store image in here
    imLoad = []

    # numbers of files found in directories
    expCat = 0  # expected cat
    expDog = 0  # expected dog

    directories = ["eval/cat/", "eval/dog/"]

    # read files in folders cat dog
    for f in os.listdir(directories[0]):
        # /eval/cat/ + fileName
        fullName = directories[0] + f
        imLoad.append(fullName)
        expCat += 1

    for f in os.listdir(directories[1]):
        fullName = directories[1] + f
        imLoad.append(fullName)
        expDog += 1

    numCat = 0
    numDog = 0

    # for img, name in imLoad:
    for name in imLoad:
        loadedImage = cv2.imread(name)

        if (loadedImage is not None):
            img = cv2.resize(loadedImage, (150, 150))
        else:
            print("??????????????ERROR???????????????")

        print("\n===", name, "===")
        # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray_image = gray_image.reshape((1,) + gray_image.shape + (1,))
        gray_image = img.reshape((1,) + img.shape)
        # print('prediction of [1, 1]: ', loaded_model.predict_classes(img, verbose=verbose))

        y_proba = loaded_model.predict(gray_image)
        y_classes = loaded_model.predict_classes(gray_image)

        print("proba:", y_proba)
        print("classe:", y_classes)
        # print(loaded_model.metrics_names)

        if y_classes == 0:
            numCat += 1
        elif y_classes == 1:
            numDog += 1

    print("cat expected:", expCat)
    print("cat found:", numCat)
    print("dog expected:", expDog)
    print("dog found:", numDog)


# eval the given file
def one(file):
    # load and display file
    loadedImage = cv2.imread(file)
    cv2.imshow("img", loadedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if (loadedImage is not None):
        img = cv2.resize(loadedImage, (150, 150))
    else:
        print("??????????????ERROR???????????????")

    print("===", file, "===")
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_image = gray_image.reshape((1,) + gray_image.shape + (1,))
    gray_image = img.reshape((1,) + img.shape)
    # print('prediction of [1, 1]: ', loaded_model.predict_classes(img, verbose=verbose))


    # seem to be: 0 cats, 1 dogs
    y_proba = loaded_model.predict(gray_image)
    print(y_proba)
    y_classes = loaded_model.predict_classes(gray_image, verbose=0)
    print(y_classes)

    if y_classes == 0:
        print("It's a cat!")
    elif y_classes == 1:
        print("It's a dog!")


def loadModel(modelName):
    # load json and create model
    # json_file = open('model.json', 'r')
    json_file = open(modelName + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    # loaded_model.load_weights("model.h5")
    loaded_model.load_weights(modelName + "/model.h5")
    print("Loaded model from disk")

    img_width, img_height = 150, 150
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return loaded_model


def testSomeFiles():
    list = ["cat.12497.jpg", "cat.12498.jpg", "cat.12499.jpg",
            "dog.12497.jpg", "dog.12498.jpg", "dog.12499.jpg"]

    # one(os.getcwd() + "\\" + list[0])  # marche pas avec le 2eme model
    # one(os.getcwd()+"\\"+list[1])
    # one(os.getcwd()+"\\"+list[2]) # ne sait pas ce que c'est
    # one(os.getcwd()+"\\"+list[3]) # marche pas avec le model1
    # one(os.getcwd()+"\\"+list[4]) # faux
    # one(os.getcwd()+"\\"+list[5])
    # one("chat.jpg")
    # one("dog1.jpg")


if __name__ == "__main__":
    # Get args
    print(sys.argv)
    modelName = sys.argv[1]
    if len(sys.argv) < 3:
        imageFile = None
    else:
        imageFile = sys.argv[2]

    # loaded_model = 0

    # Load images
    if (modelName is not None):
        loaded_model = loadModel(modelName)
    else:
        print("No model given. Please give the absolute path to your model")

    if (imageFile is not None):
        one(imageFile)
    else:
        # eval the files in /eval/ directories
        multipleEval()
