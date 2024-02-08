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
#from harvesterCv import Adquisition
import shutil

# numEval = 0
# cam = 0


class Classify():
    def __init__(self,nameId,categories = ['OK','NG','Empty']):
        self.nameId = nameId
        self.categories = categories
        self.model = []
        self.class_names = []
        self.data_dir = Path('../Models/',Path(self.nameId).with_suffix(''))
        self.batch_size = 8#32
        self.img_height = 180
        self.img_width = 180
        self.epochs=15
        self.checkpoint_path = Path(self.data_dir,"cp.ckpt")
        self.checkFolder()

    def train(self):
        # global class_names
        #okPics = list(data_dir.glob('MTG Red Man/*'))
        #nokPics = list(data_dir.glob('MTG White Mana/*'))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
        
        self.createDatasets()
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        

        self.createModel()
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=[cp_callback])  # Pass callback to training
        
        # return self.model
    def createDatasets(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        self.class_names = self.train_ds.class_names

    def createModel(self):
        num_classes = len(self.class_names)
        self.model = tf.keras.Sequential([
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

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        

    def rnEval(self,img):
        img= cv2.resize(img,dsize=(self.img_height,self.img_width), interpolation = cv2.INTER_CUBIC)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        #print(predictions)
        #print(score)

        return score
    
    
    def saveImg(self,img,category,idName = ''):
        self.checkFolder()
        tempPath = Path(self.data_dir,self.categories[category])
        ls = os.listdir(tempPath)
        ls.sort()
        #print(ls)
        if(ls == []):
            num = 0
        else:
            num = int(ls[-1][-7:-4])+1
        cv2.imwrite(str(tempPath)+'/img' + idName + f"{num:03}" +'.png',img)
        return len(ls)


    def checkFolder(self):
        for c in self.categories:
            if(not Path(self.data_dir,c).is_dir()):
                os.makedirs(Path(self.data_dir,c))

    def removeFolder(self):
        shutil.rmtree(self.data_dir)

    def saveModel(self):
        pass
        #TF lite
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        # tftlite_model = converter.convert()
        # #save model
        # with open(str(self.data_dir) + '/' + self.nameId + '_model.tflite', 'wb') as f:
        #     f.write(tftlite_model)

    def reloadModel(self):
        #self.model
        self.createDatasets()
        self.createModel()
        # Loads the weights
        self.model.load_weights(self.checkpoint_path)

        # Re-evaluate the model
        # loss, acc = self.model.evaluate(test_images, test_labels, verbose=2)
        # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
        



def test_run():
    c = cv2.VideoCapture(1)

    trained = False
    scale_percent = 100
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
        if key == ord('s')&0xFF:
            clas.saveModel()
        elif(key > 48 and key < 52):
            clas.saveImg(img0,key-49)
            title = "saved " + clas.categories[key-49]
    cv2.destroyAllWindows()
    c.release()





# if __name__ == '__main__':
#     test_run()
