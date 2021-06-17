from flask import Flask, request, jsonify
import numpy as np
import requests
import tensorflow as tf

model = tf.keras.models.load_model("model/covid_xray_analysis.h5")

abc = {'COVID19': 0, 'NORMAL': 1, 'PNEUMONIA': 2}

app = Flask(__name__)


@app.route('/', methods=['GET'])
def prediction():
    try:
        link = request.args['link']
        print(link)
        file_path = link
        content = requests.get(file_path).content
        content = tf.image.decode_jpeg(content, channels=3)
        content = tf.cast(content, tf.float32)
        content /= 255.0
        content = tf.image.resize(content, [150, 150])
        content = np.expand_dims(content, axis=0)
        content = model.predict(content).round(3)
        content = np.argmax(content)
        for key in abc:
            if content == abc[key]:
                hj = key
                com = {'predict': hj}
                return jsonify(com)

        return 'hello'
    except KeyError:
        return 'Not Working'


if __name__ == '__main__':
    app.debug = True
    app.run(port=2000)
