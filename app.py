from flask import render_template,request,jsonify,make_response,json
from controller import create_data as cd, face_recognize as fr
from flask_api import FlaskAPI

app = FlaskAPI(__name__)
@app.route('/')
def index(name=None):
    return make_response(jsonify({"message":"Application Running....."}))

@app.route('/addimage/<path>',methods=['POST'])
def create(path):
    payload = request.get_json(force=True)
    # print(payload)
    img = payload.get('image_binary')
    app_path = path
    user = payload.get('user')
    newImage = cd.create_image(img,app_path,user)
    return make_response(jsonify({"data":"Created"}),200)

@app.route('/train/<path>',methods=['GET'])
def train(path):
    training = fr.train(path)
    return make_response(jsonify({"data":"Trained"}),200)

@app.route('/fetch/<path>',methods=['POST'])
def gettrained(path):
    payload = request.get_json(force=True)
    # print(payload)
    img = payload.get('image_binary')
    result = fr.gettrained(path,img)

    if 'error' in result.keys():
        return make_response(jsonify(result),500)
    else:
        return make_response(jsonify(result),200)
   
if __name__ == "__main__":
    app.run(host='87.106.171.232',port=3000,threaded=True)