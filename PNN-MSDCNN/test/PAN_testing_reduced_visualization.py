
import tensorflow as tf
import numpy as np
from PNN_test import PNN_test
import scipy.ndimage
from scipy.io import loadmat
from RMSE import SAM,RMSE, ERGAS,QAVE
from downgraded import downgraded
from imagequantile import image_quantile, image_stretch
import matplotlib.pyplot as plt

import os
from skimage.external import tifffile

from read_image_real import read_image_ms,read_image_pan


###load image

PAN_test_data,img_MS=read_image_pan('D:\\Doktora_tez\\paper_project\\PS_paper\\test\\worldview3\\test_sim\\PAN')
MS_test_data,img_PAN=read_image_ms('D:\\Doktora_tez\\paper_project\\PS_paper\\test\\worldview3\\test_sim\\MS')



n = len(PAN_test_data)

ratio=4
params = loadmat('C:/Tools/Python/Python35/RemoteSensingEx/myCodes/training/newdata/300_070964net_1203A.mat')



for i in range(1,100):
           print(i)
           print(img_MS[i])
           print(img_PAN[i])
           Pan_data=np.expand_dims(PAN_test_data[i,:,:],axis=0)
           MS_data = np.squeeze(MS_test_data[i,:,:,:]).transpose(2,0,1)
           

###testing

           MS_HR = PNN_test(MS_data,Pan_data,params,mode='reduced')


         
           

###visualization
           plt.close('all')

           plt.ion()
           plt.figure()

           th_PAN = image_quantile(Pan_data,np.array([0.01,0.99]))
           PAN=image_stretch(np.squeeze(Pan_data),np.squeeze(th_PAN))
           plt.imshow( image_stretch(np.squeeze(Pan_data),np.squeeze(th_PAN)),cmap='gray',clim=[0,1])
           plt.title('PANCHROMATIC'), plt.axis('off')


           RGB_indexes=np.array([3,2,1])
           RGB_indexes = RGB_indexes - 1


           plt.figure()
           th_MSrgb=image_quantile(np.squeeze(MS_data[RGB_indexes,:,:]), np.array([0.01, 0.99]));
           d=image_stretch(np.squeeze(MS_data[RGB_indexes,:,:]),th_MSrgb)
           d[d<0]=0
           d[d>1]=1


           plt.imshow(d.transpose(1,2,0))
           plt.title('MULTISPECTRAL LOW RESOLUTION'), plt.axis('off')



           plt.figure()
           MS_HR = np.squeeze(MS_HR)
##           print(MS_data)
           
           
##           print(np.squeeze(MS_HR[:,:,3]).shape)

           plt.hist(np.squeeze(MS_HR[:,:,3]),bins = 5,alpha=0.5)
           

           plt.figure()
           MS_HR = MS_HR.transpose(2,0,1)
##           print(MS_HR)
           c=image_stretch(np.squeeze(MS_HR[RGB_indexes,:,:]),th_MSrgb)
           c[c<0]=0
           c[c>1]=1
           plt.imshow(c.transpose(1,2,0))

           plt.title('Pansharpened'), plt.axis('off')
           
           plt.ioff()
           plt.show()


