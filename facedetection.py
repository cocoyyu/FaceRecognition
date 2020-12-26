#!/usr/bin/env python
# coding: utf-8



import zipfile
from PIL import Image
import pytesseract
import cv2 as cv
import numpy as np

# loading the face detection classifier
# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
#install tesseract to your local
pytesseract.pytesseract.tesseract_cmd = r'D:\downloaded programs\tesseract\tesseract.exe'





class Facerecognition:
    def __init__(self):
        self.map={}
        
    #unzip the zipfile and load the images 
    def loadimage(self,path):
        #path="D:\Anaconda\Anaconda\envs\py3env\lib\small_copy.zip"
        z = zipfile.ZipFile(path, 'r')
        for f in z.infolist(): 
            ifile=z.open(f)
            image=Image.open(ifile).convert('RGB')
            text = pytesseract.image_to_string(image).replace('\n', ' ')
            #img = cv.imread(ifile)
            #gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_pil=np.array(image)
            img_cv=cv.cvtColor(img_pil,cv.COLOR_RGB2BGR)
            faces=face_cascade.detectMultiScale(img_cv,1.3,5)
            faces_bx=[]
            for x,y,w,h in faces:
                faces_bx.append((x,y,x+w,y+h))
            self.map[f.filename]=[[text],faces_bx,image]
            
    # search the target word and start face detection if the target word is in the image       
    def wordsearch(self,word):
        for filename,value in self.map.items():
            if word not in value[0][0]:
                continue
            print("Result found in file "+filename)    
            res=self.facedetect(value[1],value[2])
            if not res:
                print('But there were no faces in that file')
            else:
                display(res)
                
    # find the faces in images based on bounding boxes and paste them together            
    def facedetect(self,boundingbox,image):
        x,y=0,0
        image_face=[]
        for bx in boundingbox:
            image_face.append(image.crop(bx))
        if not image_face:
            return None
        contact_sheet=Image.new('RGB',(500, 100*(-(-len(image_face)//5))))      
        for face in image_face:
            face.thumbnail((100, 100))
            contact_sheet.paste(face, (x, y) )
            if x+100 == 500:
                x=0
                y=y+100
            else:
                x=x+100
        return contact_sheet


    


#if __name__=='__main__':
# test the main fuction
test=Facerecognition()



test.loadimage("small_img.zip")


test.wordsearch('Christopher')



