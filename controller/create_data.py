#creating database
import cv2, sys, numpy as np, os
from PIL import Image
import io
import base64

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def create_image(img,app_dir,user):
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = app_dir  #All the faces data will be present this folder
    sub_data = user     #

    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(app_dir):
        os.mkdir(app_dir)
    if not os.path.isdir(path):
        os.mkdir(path)
    (width, height) = (130, 100)    # defining the size of image


    face_cascade = cv2.CascadeClassifier(haar_file)
    count = 0
    for base, dirs, files in os.walk(path):
        for Files in files:
            count += 1
    im = np.array(stringToImage(img))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        print('%s/%s.png' % (path,count))
        cv2.imwrite('%s/%s.png' % (path,count), face_resize)
    count += 1