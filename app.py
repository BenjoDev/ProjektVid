from flask import Flask, jsonify, request
import werkzeug
from newFaceRecognition import *
import glob
import os

app = Flask(__name__)

@app.route('/upload', methods = ['POST'])
def upload():
    print("test")
    global name
    if(request.method == 'POST'):
        imageFile = request.files['image']
        filename = werkzeug.utils.secure_filename(imageFile.filename)
        imageFile.save("./uploaded_images/" + filename)
        message = face_recognition("./uploaded_images/" + filename)
        removing_files = glob.glob('./uploaded_images/*.jpg')
        for i in removing_files:
            os.remove(i)
        return jsonify({
            "message": str(message)
        })

if __name__ == "__main__":
    app.run(port=5000, debug=True, host='0.0.0.0')