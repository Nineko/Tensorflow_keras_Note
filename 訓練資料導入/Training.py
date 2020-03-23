from keras import layers,models,optimizers
from keras.engine.saving import load_model
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import math
import random
from random import shuffle
from NET.Combine.Net_Autoencoder import AE,cross_entropy_balanced
from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator,apply_affine_transform
from keras.utils.data_utils import Sequence

#PKL訓練資料讀取
PKL_Dir = 'TrainingData/PKL/AE_16560.pkl'
train = pd.read_pickle(PKL_Dir)
#讀取已訓練權重
load = True
Load_Name = 'Weight/AE.h5'
#儲存訓練權重名稱
Save_Name = 'Weight/AE.h5'  

#為輸入資料加入高斯模糊(1-7size)
def image_blur(Image_list):
    batch = Image_list.shape[0]
    blur_image = []
    for i in range(batch) :
     kernel_size = random.randint(1, 7)
     if kernel_size % 2 != 1:
      kernel_size += 1
     img_temp = cv2.GaussianBlur(Image_list[i], (kernel_size, kernel_size), 0) 
     blur_image.append(img_temp)
    blur_image = np.array(blur_image)
    return blur_image

#建構訓練資料Batch時的讀取圖像
def load_image(im,color_mode='grayscale'): 

   img = img_to_array(load_img(im,color_mode=color_mode)) / 255.
   #若需要添加隨機 10 pixel的x-y方向位移
   #img_shift = apply_affine_transform(img,tx=random.randint(1,10), ty=random.randint(1,10),fill_mode='nearest')
   return img

#讀取PKL並進行訓練Batch的產生
class DataSequence(Sequence):

    def __init__(self, df, batch_size, mode='train'):
        self.df = df
        self.bsz = batch_size
        self.mode = mode 
        self.Wireframe_list = self.df['Wireframe'].tolist()
        self.Model_list = self.df['Model'].tolist()
        self.indexes = np.arange(len(self.df['Model'].tolist()))
    def __len__(self):

        return int(math.ceil(len(self.df) / float(self.bsz)))

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.Model_list))
        if self.mode == 'train':
         np.random.shuffle(self.indexes)

    def get_batch_Wireframes(self, idx):

        Batch_indexes = self.indexes[idx*self.bsz:(idx+1)*self.bsz]
        Answer_list_temp = [self.Wireframe_list[k] for k in Batch_indexes]
        
        return np.array([load_image(im,color_mode='grayscale') for im in Answer_list_temp])

    def get_batch_Models(self, idx):

        Batch_indexes = self.indexes[idx*self.bsz:(idx+1)*self.bsz]
        Train_list_temp = [self.Model_list[k] for k in Batch_indexes]
        batch_model = np.array([load_image(im,color_mode='rgb') for im in Train_list_temp])
        batch_blur_model = image_blur(batch_model)
        return batch_blur_model

    def __getitem__(self, idx):

        batch_model = self.get_batch_Models(idx)
        batch_wireframe = self.get_batch_Wireframes(idx)
 
        return batch_model,batch_wireframe

#進行模型的初始化 
model = AE()

if load :
 model.load_weights(Load_Name,by_name = True)

#進行模型的檢視
model.summary()

#進行模型的編譯,加入Loss function及優化器
model.compile(loss={'Wireframe_out' : cross_entropy_balanced
                   }, 
              optimizer=optimizers.Adam(lr=1e-5)
             )

#產生Batch資料
seq = DataSequence(train,batch_size=25)

#進行模型的訓練
history = model.fit_generator(seq,
                              steps_per_epoch=663,
                              epochs=20
                              )

#訓練完模型儲存
model.save_weights(Save_Name)

#利用plt繪製訓練時的 loss 曲線
loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'k', label='Total loss')
plt.title('Training loss')
plt.legend()
plt.show()

