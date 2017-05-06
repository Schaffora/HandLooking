from keras.models import model_from_json
import keras
import numpy as np
import cv2
import os


def multipleEval():
    imLoad = []

    expCat = 0
    expDog = 0
    directories = ["/eval/cat/", "/eval/dog/"]
    # read files in folders cat dog

    for f in os.listdir(os.getcwd() + directories[0]):
        # imLoad.append([cv2.imread(f), f])
        fullName = os.getcwd() + directories[0] + f
        imLoad.append(fullName)
        expCat += 1

    for f in os.listdir(os.getcwd() + directories[1]):
        fullName = os.getcwd() + directories[1] + f
        # imLoad.append([cv2.imread(f), f])
        imLoad.append(fullName)
        expDog += 1

    numCat = 0
    maybeCat = 0
    numDog = 0
    maybeDog = 0

    # for img, name in imLoad:
    for name in imLoad:
        loadedImage = cv2.imread(name)
        # cv2.imshow(name, loadedImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if (loadedImage is not None):
            img = cv2.resize(loadedImage, (150, 150))
        else:
            print("??????????????ERROR???????????????")

        print("\n===", name, "===")
        # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray_image = gray_image.reshape((1,) + gray_image.shape + (1,))
        gray_image = img.reshape((1,) + img.shape)
        # print('prediction of [1, 1]: ', loaded_model.predict_classes(img, verbose=verbose))


        # seem to be: 1 cats, 0 dogs
        y_proba = loaded_model.predict(gray_image)
        print("proba:",y_proba)
        y_classes = loaded_model.predict_classes(gray_image)
        print("classe:",y_classes)

        if y_classes == 0:
            # if y_proba[0][0] < 0.5:
            #     maybeDog += 1
            # else:
            numCat += 1
        elif y_classes == 1:
            # if y_proba[0][0] < 0.5:
            #     maybeCat += 1
            # else:
            numDog += 1
            # y_classes = keras.np_utils.probas_to_classes(y_proba)

    print("cat expected:", expCat)
    print("cat found:", numCat)
    print("dog expected:", expDog)
    print("dog found:", numDog)
    print("maybe dog:", maybeDog)
    print("maybe cat:", maybeCat)


def one(file):
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


    # seem to be: 1 cats, 0 dogs
    y_proba = loaded_model.predict(gray_image)
    print(y_proba)
    y_classes = loaded_model.predict_classes(gray_image)
    print(y_classes)

    if y_classes == 0:
        print("It's a cat!")
    elif y_classes == 1:
        print("It's a dog!")


if __name__ == "__main__":
    Choices = ["modelWith10k/", ""]
    modelChoice = Choices[0] #0-1 to change model

    # load json and create model
    # json_file = open('model.json', 'r')
    json_file = open(modelChoice + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    # loaded_model.load_weights("model.h5")
    loaded_model.load_weights(modelChoice + "model.h5")
    print("Loaded model from disk")

    img_width, img_height = 150, 150
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    verbose = 0

    #eval the files in /eval/ directories
    multipleEval()

    list = ["cat.12497.jpg", "cat.12498.jpg", "cat.12499.jpg",
            "dog.12497.jpg", "dog.12498.jpg", "dog.12499.jpg"]

    # one(os.getcwd() + "\\" + list[0])  # marche pas avec le 2eme model
    # one(os.getcwd()+"\\"+list[1])
    # one(os.getcwd()+"\\"+list[2]) # ne sait pas ce que c'est
    # one(os.getcwd()+"\\"+list[3]) # marche pas avec le model1
    # one(os.getcwd()+"\\"+list[4]) # faux
    # one(os.getcwd()+"\\"+list[5])
    # one(os.getcwd()+"\\"+"chat.jpg")

