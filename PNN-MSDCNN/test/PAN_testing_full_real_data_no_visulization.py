
import tensorflow as tf
import numpy as np
from PNN_test_resnet import PNN_test

from scipy.io import loadmat,savemat
from RMSE import SAM,RMSE, ERGAS,QAVE

import os
from skimage.external import tifffile

from read_image_real import read_image_ms,read_image_pan
from imagequantile import image_quantile, image_stretch
import matplotlib.pyplot as plt

###load image

MS_test_data,img_MS=read_image_ms('C:/Users/USER-1/Documents/Peijuan/imageFromCenter/testing1211/test_real/MS')
PAN_test_data,img_PAN=read_image_pan('C:/Users/USER-1/Documents/Peijuan/imageFromCenter/testing1211/test_real/PAN')






n = len(PAN_test_data)
ratio = 4
m = PAN_test_data[0,:,:].shape[0]
k = PAN_test_data[0,:,:].shape[1]

MS_HR_All=np.empty([50,m,k,ratio])

params = loadmat(
           'C:/Tools/Python/Python35/RemoteSensingEx/myCodes/training/params/MSDCNN_indices_L2.mat')


for i in range(50,100):
           print('number:',i)
#Pan_data=np.expand_dims(np.double(Pan_data),axis=0)
#MS_data=np.squeeze(np.double(MS_data)).transpose(2,0,1)
           print(img_MS[i])
           print(img_PAN[i])
           
           Pan_data=np.expand_dims(PAN_test_data[i,:,:],axis=0)
           MS_data=np.squeeze(MS_test_data[i,:,:,:]).transpose(2,0,1)

###testing


           MS_HR = PNN_test(MS_data,Pan_data,params,mode='full')



           MS_HR = np.squeeze(MS_HR)
           #MS_HR = MS_HR.transpose(2,0,1)
           

           MS_HR_All[i-50,:,:,:]=MS_HR

           
           
savemat('MS_HR_100P.mat',mdict={'MS_HR_100P':MS_HR_All})           







