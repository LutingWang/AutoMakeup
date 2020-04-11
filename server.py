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
for i in range(3):
    with Image.open(f'refs/{i}.png') as image:
        _, prep = makeup.preprocess(image)
        refs.append(prep)

test_result = Image.open('assets/test_result.png')
anchor = (200, 500)

app = Flask(__name__)


@app.route('/transfer/', methods = ['POST'])
def transfer():
    data = json.loads(request.get_data().decode())
    model = data.get('model')
    image = futils.fpp.decode(data.get('file'))
    box, prep = makeup.preprocess(image)
    result = makeup.solver.test(*prep, *refs[model])
    result = futils.fpp.beautify(result) # base64
    result = futils.fpp.decode(result)
    result = futils.merge(image, result, box)
    return futils.fpp.encode(result)


@app.route('/exchange/', methods=['POST'])
def exchange():
    data = json.loads(request.get_data().decode())
    images = [futils.fpp.decode(image) for image in data]
    boxes, preps = [], []
    for image in images:
        box, prep = makeup.preprocess(image)
        boxes.append(box)
        preps.append(prep)
    results = [makeup.solver.test(*preps[0], *preps[1]), 
              makeup.solver.test(*preps[1], *preps[0])]
    for i in range(2):
        results[i] = futils.fpp.beautify(results[i]) # base64
        results[i] = futils.fpp.decode(results[i])
        results[i] = futils.merge(images[i], results[i], boxes[i])
        results[i] = futils.fpp.encode(results[i])
    return json.dumps(results)


@app.route('/test/', methods=['POST'])
def test():
    image = json.loads(request.get_data().decode()).get('file')
    src_score = futils.fpp.rank(image)
    image = futils.fpp.decode(image)
    _, prep = makeup.preprocess(image)
    
    model_id = -1
    max_score = src_score
    result = image
    for i, model in enumerate(refs):
        temp = makeup.solver.test(*prep, *model)
        score = futils.fpp.rank(temp)
        if score > max_score:
            model_id = i
            max_score = score
            result = temp
    score = (max_score - src_score) / (100 - src_score) * 100

    result = futils.fpp.beautify(result) # base64
    result = futils.fpp.decode(result)
    bg = test_result.copy()
    bg.paste(result, box=anchor)
    # return {
    #     'file': futils.fpp.encode(bg),
    #     'score': score,
    #     'id': model_id,
    #     }
    return futils.fpp.encode(bg)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5001, debug = True, ssl_context = ('ssl/server.crt', 'ssl/server.key'))
    # src = Image.open('refs/0.png').convert('RGB')
    # ref = Image.open('refs/2.png').convert('RGB')
    # result = makeup.solver.test(*(makeup.preprocess(src)), *(makeup.preprocess(ref)))
    # result.save('result.png')
