import cv2
import numpy as np
from keras.models import load_model  

def face_recognition(pot_slike):
    name = "other"
    np.set_printoptions(suppress=True)

    model = load_model("keras_Model.h5", compile=False)

    image = cv2.imread(pot_slike)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)

    if(index == 1):
        name = "benjamin"
    elif(index == 2):
        name = "zan"
    return name # 0 = other, 1 = Benjamin, 2 = Zan
