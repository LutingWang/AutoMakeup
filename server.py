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
        refs.append(makeup.preprocess(image))

app = Flask(__name__)


@app.route('/transfer/', methods = ['POST'])
def transfer():
    data = json.loads(request.get_data().decode())
    model = data.get('model')
    image = base64.b64decode(data.get('file'))
    image = Image.open(BytesIO(image))
    image = makeup.solver.test(*(makeup.preprocess(image)), *(refs[model]))
    return futils.fpp.beautify(image)


@app.route('/exchange/', methods=['POST'])
def exchange():
    data = json.loads(request.get_data().decode())
    images = [base64.b64decode(image) for image in data]
    images = [Image.open(BytesIO(image)) for image in images]
    images = [makeup.preprocess(image) for image in images]
    images = [makeup.solver.test(*(images[0]), *(images[1])), 
              makeup.solver.test(*(images[1]), *(images[0]))]
    images = [futils.fpp.beautify(image) for image in images]
    return json.dumps(images)


@app.route('/test/', methods=['POST'])
def test():
    image = json.loads(request.get_data().decode()).get('file')
    src_score = futils.fpp.rank(image)
    image = Image.open(BytesIO(base64.b64decode(image)))
    max_score = src_score
    result = image
    for model in refs:
        temp = makeup.solver.test(*(makeup.preprocess(image)), *(model))
        score = futils.fpp.rank(temp)
        if score > max_score:
            max_score = score
            result = temp
    return { 'file': futils.fpp.beautify(result), 'score': score }


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5001, debug = True, ssl_context = ('ssl/server.crt', 'ssl/server.key'))
    # src = Image.open('refs/0.png').convert('RGB')
    # ref = Image.open('refs/2.png').convert('RGB')
    # result = makeup.solver.test(*(makeup.preprocess(src)), *(makeup.preprocess(ref)))
    # result.save('result.png')
