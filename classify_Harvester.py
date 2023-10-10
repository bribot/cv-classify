import numpy as np
import cv2
from harvesterCv import Adquisition

from classify import Classify


def main():
    
    
    c = Adquisition()
    c.getDevices()
    c.setAdquisitionDevice({'serial_number': 'S1176510'})
    c.setExposure(50000)
    c.setGain(1)
    c.startAdquisition()


    trained = False
    scale_percent = 20
    ret,frame = c.read()
    height, width, layer = frame.shape
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame,dim,interpolation=cv2.INTER_NEAREST)
    #img_height,img_width,l = frame.shape
    roi0 = cv2.selectROI(frame)
    cv2.destroyAllWindows()
    title = "Feed"
    clas = Classify('test',categories=['oki','noki','doki'])
    while(True):
        # global model
        ret,frame = c.read()
        # resize image
        img = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img0 = img[roi0[1]:roi0[1]+roi0[3], roi0[0]:roi0[0]+roi0[2]]
        cv2.rectangle(img,roi0,color=(0,0,255))
        if(trained):
            score = clas.rnEval(img0)
            title = clas.class_names[np.argmax(score)]

        cv2.putText(img,title,(50,100),cv2.FONT_HERSHEY_DUPLEX,3,color=(0,200,255))
        cv2.imshow('feed',img)

        key = cv2.waitKey(10)
        if key == ord('q')&0xFF:
            break
        if key == ord('r')&0xFF:
            trained = False
            clas.removeFolder()
        if key == ord('t')&0xFF:
            title = "trained!"
            clas.train()
            trained = True
        elif(key > 48 and key < 52):

            clas.saveImg(img0,key-49)
            title = "saved " + clas.categories[key-49]
            
        

        
    cv2.destroyAllWindows()
    c.release()





if __name__ == '__main__':
    main()