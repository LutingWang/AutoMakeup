from io import BytesIO

import base64
import json
from PIL import Image
from flask import Flask, request, Response

from makeup.transfer import preprocess, solver


refs = []
for i in range(1):
    with Image.open(f'refs/{i}.png') as image:
        refs.append(preprocess(image))

app = Flask(__name__)

@app.route('/transfer/', methods = ['POST'])
def transfer():
    data = json.loads(request.get_data().decode())
    model = data.get('model')
    image = base64.b64decode(data.get('file'))
    image = Image.open(BytesIO(image))
    output_buf = BytesIO()
    solver.test(*(preprocess(image)), *(refs[model])).save(output_buf, format='JPEG')
    return base64.b64encode(output_buf.getvalue())

@app.route('/exchange/', methods=['POST'])
def exchange():
    data = json.loads(request.get_data().decode())
    images = [base64.b64decode(image) for image in data]
    images = [preprocess(image) for image in images]
    result = [solver.test(*(images[0]), *(images[1])), 
              solver.test(*(images[1]), *(images[0]))]



#@app.route('/login/', methods = ['GET'])
#def login():
#    code = request.args['code']
#    openid = wx.login(code)['openid']
#    user.register_user(openid)
#    return openid
#
#@app.route('/transfer/upload/', methods = ['POST'])
#def transfer_upload():
#    openid = request.args['openid']
#    img = request.files.get('img')
#    path = user.root(openid) + 'transfer.jpg'
#    img.save(path)
#    img.close()
#    req = requests.get('http://localhost:5011/makeuptransfer/', params = { 'path': path })
#    path = req.text
#    return path[path.rindex('/') + 1:]
#
#@app.route('/transfer/', methods = ['POST'])
#def transfer():
#    data = json.loads(request.get_data().decode())
#    model = data.get('model')
#    image = base64.b64decode(data.get('file'))
#    print(model)
#    
#    with open('../BeautyGAN-100/data/makeup/test_S/src.png', 'wb') as f:
#        f.write(image)
#    with open('../Mask/face-parsing.PyTorch/test/src.png','wb') as f:
#        f.write(image)
#    for img in os.listdir("./images"):
#        # print(img)
#        if img == str(model) + ".png" :
#            shutil.copy(os.path.join("./images",img),"../Mask/face-parsing.PyTorch/test/ref.png")
#            shutil.copy(os.path.join("./images",img),"../BeautyGAN-100/data/makeup/test_R/ref.png")
#            # print("debug",img)
#            break
#
#    os.system('cd ../Mask/face-parsing.PyTorch/ && python test.py')
#    os.system('cd ..')
#    shutil.copy("../Mask/face-parsing.PyTorch/res/test_res/src.png","../BeautyGAN-100/data/makeup/test_S_mask/src.png")
#    shutil.copy("../Mask/face-parsing.PyTorch/res/test_res/ref.png","../BeautyGAN-100/data/makeup/test_R_mask/ref.png")
#
#    os.system('cd ../BeautyGAN-100/ && sh my_test.sh')
#    image_path = '../BeautyGAN-100/visulization/_spade_skin_0.1_atten_wvisual1e-2_w200_nc1_rec10_noconv_realsimple_idt1/030_testB_vFG87_usenoeye/62_2520_0_all.png'
#    with open(image_path,'rb') as f:
#        image = base64.b64encode(f.read())
#    resp = Response(image,mimetype="image/png")
#    return resp
#
#@app.route('/exchange/', methods=['POST'])
#def exchange():
#    # img1 = request.files.get('img1')
#    # img2 = request.files.get('img2')
#    # img1.save('../BeautyGAN-100/data/makeup/test_S/src.png')
#    # img2.save('../BeautyGAN-100/data/makeup/test_R/ref.png')
#    # os.system('cd ../BeautyGAN-100/ && sh my_test.sh')
#    # # img = cv2.imread('../BeautyGAN-100/visulization/_spade_skin_0.1_atten_wvisual1e-2_w200_nc1_rec10_noconv_realsimple_idt1/030_testB_vFG87_usenoeye/62_2520_0_all.png')
#    data = request.get_data()
#    images = [base64.b64decode(image) 
#              for image in json.loads(data.decode())]
#    with open('../BeautyGAN-100/data/makeup/test_S/src.png', 'wb') as f:
#        f.write(images[0])
#    with open('../BeautyGAN-100/data/makeup/test_R/ref.png', 'wb') as f:
#        f.write(images[1])
#    os.system('cd ../BeautyGAN-100/ && sh my_test.sh')
#    image_path = '../BeautyGAN-100/visulization/_spade_skin_0.1_atten_wvisual1e-2_w200_nc1_rec10_noconv_realsimple_idt1/030_testB_vFG87_usenoeye/62_2520_0_all.png'
#    with open(image_path,'rb') as f:
#        image = f.read()
#    resp = Response(image,mimetype="image/png")
#    return resp
#
#
#@app.route('/test/', methods=['GET'])
#def visdom():
#    image_path = '../BeautyGAN-100/visulization/_spade_skin_0.1_atten_wvisual1e-2_w200_nc1_rec10_noconv_realsimple_idt1/030_testB_vFG87_usenoeye/62_2520_0_all.png'
#    with open(image_path,'rb') as f:
#        image = f.read()
#    resp = Response(image,mimetype="image/png")
#    return resp
#    # return 'https://www.lutingwang.xyz:5001/visdom/'

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5001, debug = True, ssl_context = ('ssl/server.crt', 'ssl/server.key'))


