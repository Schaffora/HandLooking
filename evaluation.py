from keras.models import model_from_json
import numpy as np
from PIL import Image

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

img = Image.open('test.jpg')
img = img.convert('RGB')
img = img.resize((img_width, img_height), Image.ANTIALIAS)
x = np.asarray(img, dtype='float32')
x = np.expand_dims(x, axis=0)

out1 = loaded_model.predict_classes(x)
print(np.argmax(out1))


