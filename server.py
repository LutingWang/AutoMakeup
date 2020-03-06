#!/usr/bin/python
# -*- encoding: utf-8 -*-
from io import BytesIO

import base64
import json
from PIL import Image
from flask import Flask, request, Response

import faceutils as futils
import makeup

refs = []
for i in range(1):
    with Image.open(f'refs/{i}.png') as image:
        # refs.append(makeup.preprocess(image))
        refs.append(None)

app = Flask(__name__)


@app.route('/transfer/', methods = ['POST'])
def transfer():
    data = json.loads(request.get_data().decode())
    model = data.get('model')
    image = base64.b64decode(data.get('file'))
    image = Image.open(BytesIO(image))
    # image = solver.test(*(preprocess(image)), *(refs[model]))
    return futils.fpp.beautify(image)


@app.route('/exchange/', methods=['POST'])
def exchange():
    data = json.loads(request.get_data().decode())
    images = [base64.b64decode(image) for image in data]
    images = [Image.open(BytesIO(image)) for image in images]
    # images = [preprocess(image) for image in images]
    # images = [solver.test(*(images[0]), *(images[1])), 
    #           solver.test(*(images[1]), *(images[0]))]
    images = [images[1], images[0]]
    images = [futils.fpp.beautify(image) for image in images]
    return json.dumps(images)


@app.route('/test/', methods=['POST'])
def test():
    image = json.loads(request.get_data().decode()).get('file')
    image = Image.open(BytesIO(base64.b64decode(image)))
    max_score = -1
    for _ in refs:
        # temp = solver.test(*(preprocess(image)), *(refs[model]))
        temp = image
        score = futils.fpp.rank(temp)
        if score > max_score:
            max_score = score
            result = temp
    return futils.fpp.beautify(result)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5001, debug = True, ssl_context = ('ssl/server.crt', 'ssl/server.key'))
