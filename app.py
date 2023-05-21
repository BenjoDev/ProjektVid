from flask import Flask, jsonify, request
import werkzeug
from face_recognition import *
import glob

response = ''
app = Flask(__name__)

@app.route('/upload', methods = ['POST'])
def upload():
    if(request.method == 'POST'):
        imageFile = request.files['image']
        filename = werkzeug.utils.secure_filename(imageFile.filename)
        imageFile.save("./uploaded_images/" + filename)
        message = face_recognition("./uploaded_images/" + filename)
        print(message[0])
        removing_files = glob.glob('./uploaded_images/*.jpg')
        for i in removing_files:
            os.remove(i)
        return jsonify({
            "message": str(message[0])
        })

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(port=5000, debug=True)