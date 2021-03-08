
import tensorflow as tf
import numpy as np
from PNN_test_no_indices import PNN_test

from scipy.io import loadmat
from RMSE import SAM,RMSE, ERGAS,QAVE

import os
from skimage.external import tifffile

from read_image_real import read_image_ms,read_image_pan
from imagequantile import image_quantile, image_stretch
import matplotlib.pyplot as plt

###load image

MS_test_data,img_MS=read_image_ms('D:\\Doktora_tez\\paper_project\\PS_paper\\test\\worldview2\\test_real/MS')
PAN_test_data,img_PAN=read_image_pan('D:\\Doktora_tez\\paper_project\\PS_paper\\test\\worldview2\\test_real/PAN')







n = len(PAN_test_data)
ratio = 4
m = PAN_test_data[0,:,:].shape[0]
k = PAN_test_data[0,:,:].shape[1]

#MS_HR_All=np.empty([n,m,k,ratio])

params = loadmat(
           'C:/Tools/Python/Python35/RemoteSensingEx/myCodes/training/params/PNN_noindices_L1.mat')


for i in range(5,15):
    print(i)
    print(img_MS[i])
    print(img_PAN[i])
#Pan_data=np.expand_dims(np.double(Pan_data),axis=0)
#MS_data=np.squeeze(np.double(MS_data)).transpose(2,0,1)
    Pan_data=np.expand_dims(PAN_test_data[i,:,:],axis=0)
    MS_data=np.squeeze(MS_test_data[i,:,:,:]).transpose(2,0,1)

###testing


    MS_HR = PNN_test(MS_data,Pan_data,params,mode='full')



    MS_HR = np.squeeze(MS_HR)
##MS_HR = MS_HR.transpose(2,0,1)
           




###visualization

    plt.close('all')

    plt.ion()
    plt.figure()

    th_PAN = image_quantile(Pan_data,np.array([0.01,0.99]))
    PAN=image_stretch(np.squeeze(Pan_data),np.squeeze(th_PAN))
    plt.imshow( image_stretch(np.squeeze(Pan_data),np.squeeze(th_PAN)),cmap='gray',clim=[0,1])
    plt.title('PANCHROMATIC'), plt.axis('off')


    RGB_indexes=np.array([3,2,1])
           # RGB_indexes=np.array([1,2,3])
    RGB_indexes = RGB_indexes - 1


           # plt.figure()
    th_MSrgb=image_quantile(np.squeeze(MS_data[RGB_indexes,:,:]), np.array([0.01, 0.99]));
           # d=image_stretch(np.squeeze(MS_data[RGB_indexes,:,:]),th_MSrgb)
           # d[d<0]=0
           # d[d>1]=1


           # plt.imshow(d.transpose(1,2,0))
           # plt.title('MULTISPECTRAL LOW RESOLUTION'), plt.axis('off')



           
    MS_HR = np.squeeze(MS_HR)
    MS_HR = MS_HR.transpose(2,0,1)
    plt.figure()
    print(MS_HR.transpose(1,2,0).shape)
    plt.hist(np.squeeze(MS_HR.transpose(1,2,0)[:,:,0]),bins = 20)

           

    plt.figure()
    c=image_stretch(np.squeeze(MS_HR[RGB_indexes,:,:]),th_MSrgb)
    c[c<0]=0
    c[c>1]=1
    plt.imshow(c.transpose(1,2,0))

    plt.title('Pansharpened'), plt.axis('off')

           # plt.figure()
           # print(c.transpose(1,2,0).shape)
           # plt.hist(np.squeeze(c.transpose(1,2,0)[:,:,0]))

    plt.ioff()
    plt.show()



