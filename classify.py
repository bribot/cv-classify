#import matplotlib.pyplot as plt
import numpy as np
#import PIL
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import os
import cv2
from pathlib import Path

numEval = 0
cam = 1

cat0 = '1'
cat1 = '2'
cat2 = '3'
cat3 = '4'
cat4 = '5'
path_cat = "./trainset/"


model = []
class_names = []
data_dir = pathlib.Path("./trainset/").with_suffix('')
path_ok = "./trainset/OK/"
path_nok = "./trainset/NOK/"
batch_size = 32
img_height = 180
img_width = 180
epochs=15


def train():
    global class_names
    #okPics = list(data_dir.glob('MTG Red Man/*'))
    #nokPics = list(data_dir.glob('MTG White Mana/*'))
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    num_classes = len(class_names)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return model

# def rnEval_fromfile(myfile,model):
#     global numEval
#     #myfile = 'white_land.png'
#     fullPath = os.path.abspath("./" + myfile)
#     test_img = tf.keras.utils.get_file(str(numEval),origin='file:\\'+fullPath)
#     numEval +=1
#     img = tf.keras.utils.load_img(
#         test_img, target_size=(img_height, img_width)
#     )
    

#     img_array = tf.keras.utils.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0) # Create a batch

#     predictions = model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])
#     print(predictions)
#     print(score)
#     return score

def rnEval(img,model):
    img= cv2.resize(img,dsize=(img_height,img_width), interpolation = cv2.INTER_CUBIC)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    #print(predictions)
    #print(score)

    return score

def saveImg(img,path,idName = ''):
    ls = os.listdir(path)
    if(ls == []):
        num = 0
    else:
        num = int(ls[-1][-7:-4])+1
    cv2.imwrite(path+'img' + idName + f"{num:03}" +'.png',img)


def checkFolder(path):
    if(not Path(path).is_dir()):
       os.makedirs(path)

def main():
    
    trained = False
    #cam = "tcp://192.168.16.130:5050"
    #windowName = 'feed'
    c = cv2.VideoCapture(cam)
    # checkFolder(path_ok)
    # checkFolder(path_nok)
    ret,frame = c.read()
    #img_height,img_width,l = frame.shape
    roi0 = cv2.selectROI(frame)
    cv2.destroyAllWindows()
    title = "feed"
    while(True):
        global model
        ret,frame = c.read()
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img0 = img[roi0[1]:roi0[1]+roi0[3], roi0[0]:roi0[0]+roi0[2]]
        cv2.rectangle(img,roi0,color=(0,0,255))
        if(trained):
            score = rnEval(img0,model)
            title = class_names[np.argmax(score)]

        cv2.putText(img,title,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,color=(100,0,255))
        cv2.imshow('feed',img)

        key = cv2.waitKey(10)
        if key == ord('q')&0xFF:
            break
        if key == ord('t')&0xFF:
            title = "trained!"
            model = train()
            trained = True
            #print('Trained!')
        # if key == ord('e')&0xFF:
        #     score = rnEval(img0,model)
        #     title = class_names[np.argmax(score)]
            #print(
            #"This image most likely belongs to {} with a {:.2f} percent confidence."
            #.format(class_names[np.argmax(score)], 100 * np.max(score)))
        elif(key > 48 and key < 58):
            tempPath = path_cat+str(key-48)+'/'
            checkFolder(tempPath)
            saveImg(img0,tempPath,str(key))
            title = "saved " + str(key-48)
            
        

        
    cv2.destroyAllWindows()
    c.release()





if __name__ == '__main__':
    main()