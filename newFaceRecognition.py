import cv2
import numpy as np
from keras.models import load_model

from keras.models import load_model  # TensorFlow is required for Keras to work
# from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
# # Load the model
# model = load_model("keras_Model.h5", compile=False)
# # Load the labels
# class_names = open("labels.txt", "r").readlines()
# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# # Replace this with the path to your image
# image = Image.open("tomaz.jpg").convert("RGB")
# # resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
# # turn the image into a numpy array
# image_array = np.asarray(image)
# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
# # Load the image into the array
# data[0] = normalized_image_array
# # Predicts the model
# prediction = model.predict(data)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]

# # Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)


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
        if(classIndex == 1):
            name = "benjamin"
        else:
            name = "zan"
        print(name)
        return name # 0 = other, 1 = Benjamin, 2 = Zan
    return 0