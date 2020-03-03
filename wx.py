import requests


url = "https://api.weixin.qq.com/sns/jscode2session"
appId = "wx38c6d1a42e5d7031"
appSecret = "1e5a21bc0906062fff9795ef56818be5"

def login(code):
    params = {
        'appid': appId,
        'secret': appSecret,
        'js_code': code,
        'grant_type': 'authorization_code'
    }
    return requests.get(url, params = params, timeout = 5, verify = False).json()

