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


def face_recognition(pot_slike, ime):
    print("test2")
    facedetect = cv2.CascadeClassifier('har.xml')
    model = load_model('keras_model.h5')

    imgOrignal = cv2.imread(pot_slike)
    faces = facedetect.detectMultiScale(imgOrignal,1.3,5)

    for x,y,w,h in faces:
        print("huh")
        crop_img=imgOrignal[y:y+h,x:x+h]
        print("1")
        img=cv2.resize(crop_img, (224,224))
        print("2")        
        img=img.reshape(1, 224, 224, 3)
        print("3")
        try:
            print("hmmm")
            prediction = model.predict(img)
        except Exception as e:
            print("kaj")
            print(e)
        print("4")
        classIndex = np.argmax(prediction,axis=-1)
        print("5")
        print("class2 " + str(classIndex[0]))
        print("class " + str(classIndex))
        print("wtf")
        return classIndex # 0 = other, 1 = Benjamin, 2 = Zan
    print("no")
    return 0

# face_recognition("benjo2.jpg", "da")

# from keras.models import load_model  # TensorFlow is required for Keras to work
# import cv2  # Install opencv-python
# import numpy as np

# def face_recognition(pot_slike, ime):

# # Disable scientific notation for clarity
#     np.set_printoptions(suppress=True)

#     # Load the model
#     model = load_model("keras_model.h5", compile=False)

#     # Load the labels
#     class_names = open("labels.txt", "r").readlines()

#     # CAMERA can be 0 or 1 based on default camera of your computer
#     # camera = cv2.VideoCapture(0)

#     # while True:
#         # Grab the webcamera's image.
#         # ret, image = camera.read()

#     image = cv2.imread(pot_slike)
#         # Resize the raw image into (224-height,224-width) pixels
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

#         # Show the image in a window
#     # cv2.imshow("Webcam Image", image)

#         # Make the image a numpy array and reshape it to the models input shape.
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

#         # Normalize the image array
#     image = (image / 127.5) - 1

#         # Predicts the model
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]

#     # Print prediction and confidence score
#     print("Class:", class_name[2:], end="")
#     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
#     print(index)
#     return index

# face_recognition("benjo2.jpg", "da")
# Listen to the keyboard for presses.
# keyboard_input = cv2.waitKey(0)

# 27 is the ASCII for the esc key on your keyboard.
# if keyboard_input == 27:
#     break

# camera.release()
# cv2.destroyAllWindows()
