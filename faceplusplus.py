# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import time
import json


http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "-fd9YqPnrLnmugQGAhQoimCkQd0t8N8L"
secret = "0GLyRIHDnrjKSlDuflLPO8a6U32hyDUy"

def detect(filepath):
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    fr = open(filepath, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s--\r\n' % boundary)
    
    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')
    
    http_body = b'\r\n'.join(data)
    
    # build http request
    req = urllib.request.Request(url=http_url, data=http_body)
    
    # header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
    
    try:
        # post data to server
        resp = urllib.request.urlopen(req, timeout=5)
        # get response
        return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(e.read().decode('utf-8'))
