#import matplotlib.pyplot as plt
import numpy as np
#import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
import os

import cv2

numEval = 0


model = []
class_names = []
data_dir = pathlib.Path("./trainset/").with_suffix('')
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
    num_classes = 2

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

def rnEval(myfile,model):
    global numEval
    #myfile = 'white_land.png'
    fullPath = os.path.abspath("./" + myfile)
    test_img = tf.keras.utils.get_file(str(numEval),origin='file:\\'+fullPath)
    numEval +=1
    img = tf.keras.utils.load_img(
        test_img, target_size=(img_height, img_width)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(predictions)
    print(score)

    return score

# if __name__ == '__main__':
#     main()

def main():
    c = cv2.VideoCapture(1)
    numOk = 0
    numNOk = 0

    title = "feed"
    while(True):
        global model
        ret,frame = c.read()
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.putText(img,title,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,color=(255,0,0))
        cv2.imshow('feed',img)
        
        if cv2.waitKey(10) == ord('q')&0xFF:
            break
        if cv2.waitKey(10) == ord('o')&0xFF:
            title = "saved OK " + str(numOk)
            print('saved OK!')
            cv2.imwrite('./trainset/OK/imgOk' + str(numOk)+'.png',img)
            numOk +=1
        if cv2.waitKey(10) == ord('n')&0xFF:
            title = "saved NOK " + str(numNOk)
            print('saved NOK!')
            cv2.imwrite('./trainset/NOK/imgNOk' + str(numNOk)+'.png',img)
            numNOk +=1
        if cv2.waitKey(10) == ord('t')&0xFF:
            title = "trained!"
            model = train()
            print('Trained!')
        if cv2.waitKey(10) == ord('e')&0xFF:
            cv2.imwrite('./eval.png',img)
            score = rnEval('eval.png',model)
            title = class_names[np.argmax(score)]
            print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score)))
            
        

        
    cv2.destroyAllWindows()
    c.release()

def main_temp():
    model = train()
    myfile = 'white_land.png'
    score = rnEval(myfile, model)
    print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

if __name__ == '__main__':
    main()