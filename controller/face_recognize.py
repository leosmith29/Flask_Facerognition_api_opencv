# facerec.py
import cv2, sys, numpy, os,pandas as pd
from . import create_data as cr
def train(app_path):

    datasets = app_path
    model_path = "model"
    identity_path = "csv_identity"

    # Create a list of images and a list of corresponding names
    (images, labels, names, id) = ([], [], [], 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names.append({"sn":id,"identity":subdir})
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
 
    # Create a Numpy array from the two lists above
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]

    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)
    model.save('%s/trainedModel_%s.yml' % (model_path,app_path))
    #save names of trained images
    pd.DataFrame(names).to_csv('%s/%s_trained_images_names.csv' % (identity_path,app_path),columns=['sn','identity'],index=False)
  
def gettrained(app_path,img):
    
    haar_file = 'haarcascade_frontalface_default.xml'
    model_path = "model"
    identity_path = "csv_identity"
    # NOTE FOR OpenCV2: remove '.face'
    model = cv2.face.LBPHFaceRecognizer_create()    
    model.read('%s/trainedModel_%s.yml' % (model_path,app_path))
    
    
    face_cascade = cv2.CascadeClassifier(haar_file)
    
    im =  numpy.array(cr.stringToImage(img))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    (width, height) = (130, 100)
    names = pd.read_csv('%s/%s_trained_images_names.csv' % (identity_path,app_path),squeeze=False,index_col=False,usecols=['sn','identity'])
 
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1]<500:
            rec = names.iloc[prediction[0]].to_dict()
            if 'identity' in rec.keys():
                return {"path":app_path,"data": rec['identity']}
            else:
                return {"error": "No identy found"}                    
        else:
            return {"error": "No identy found"}
            cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
