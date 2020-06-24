#!/usr/bin/python
# -*- encoding: utf-8 -*-
from io import BytesIO

import base64
import json
import requests

from PIL import Image

key = "-fd9YqPnrLnmugQGAhQoimCkQd0t8N8L"
secret = "0GLyRIHDnrjKSlDuflLPO8a6U32hyDUy"


def encode(image: Image) -> str:
    with BytesIO() as output_buf:
        image.save(output_buf, format='PNG')
        return base64.b64encode(output_buf.getvalue()).decode('utf-8')


def decode(image: base64) -> Image:
    image = base64.b64decode(image)
    image = Image.open(BytesIO(image))
    return image


def beautify(image: Image or str) -> str:
    if not isinstance(image, str):
        image = encode(image)
    data = {
        'api_key': key,
        'api_secret': secret,
        'image_base64': image,
        }
    resp = requests.post(beautify.url, data=data).json()
    return resp.get('result', image)


def rank(image: Image or str) -> int:
    if not isinstance(image, str):
        image = encode(image)
    data = {
        'api_key': key,
        'api_secret': secret,
        'image_base64': image,
        'return_attributes': 'beauty',
        'beauty_score_min': 71,
        }
    resp = requests.post(rank.url, data=data).json()
    if 'faces' not in resp.keys(): return 100

    scores = resp['faces'][0]['attributes']['beauty']
    return max(scores.values())


beautify.url = 'https://api-cn.faceplusplus.com/facepp/v2/beautify'
rank.url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
