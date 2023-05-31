import cv2
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work

def face_recognition(pot_slike):
    name = "other"
    facedetect = cv2.CascadeClassifier('har.xml')
    model = load_model('keras_model.h5')

    imgOrignal = cv2.imread(pot_slike)
    faces = facedetect.detectMultiScale(imgOrignal,1.3,5)

    for x,y,w,h in faces:
        crop_img=imgOrignal[y:y+h,x:x+h]
        img=cv2.resize(crop_img, (224,224))
        img=img.reshape(1, 224, 224, 3)
        prediction=model.predict(img)
        classIndex = np.argmax(prediction,axis=-1)
        print(classIndex[0])
        if(classIndex[0] == 1):
            name = "benjamin"
        elif(classIndex[0] == 2):
            name = "zan"
        return name # 0 = other, 1 = Benjamin, 2 = Zan
    return name