from flask import Flask, jsonify, request
import werkzeug
from faceRecognition import *
import glob
import json
import cv2
import joblib

app = Flask(__name__)
name = '0' # 0 = Benjamin, 1 = Žan

@app.route('/name', methods = ['POST'])
def nameRoute():
    global name
    if(request.method == 'POST'):
        request_data = request.data 
        request_data = json.loads(request_data.decode('utf-8')) 
        name = request_data['name'] 
        return " "

@app.route('/upload', methods = ['POST'])
def upload():
    global name
    if(request.method == 'POST'):
        imageFile = request.files['image']
        filename = werkzeug.utils.secure_filename(imageFile.filename)
        imageFile.save("./uploaded_images/" + filename)
        message = face_recognition("./uploaded_images/" + filename, name)
        removing_files = glob.glob('./uploaded_images/*.jpg')
        for i in removing_files:
            os.remove(i)
        return jsonify({
            "message": str(message[0])
        })

if __name__ == "__main__":
    app.run(port=5000, debug=True, host='0.0.0.0')