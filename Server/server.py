from flask import Flask, request, jsonify
import util


app = Flask(__name__)

@app.route('/classify_image', methods= ["GET", "POST"])
def classify_img():
    image_data = request.form['Image Data']

    response = jsonify(util.classifyImage(image_data))

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    app.run(port= 5000)