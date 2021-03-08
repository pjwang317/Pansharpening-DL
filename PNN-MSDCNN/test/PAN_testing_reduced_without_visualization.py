
import tensorflow as tf
import numpy as np
from PNN_test_no_indices import PNN_test
#from PNN_test_resnet_no_indices import PNN_test
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

PAN_test_data,img_PAN=read_image_pan('D:\\Doktora_tez\\paper_project\\PS_paper\\test\\worldview2\\test_sim\\PAN')
MS_test_data,img_MS=read_image_ms('D:\\Doktora_tez\\paper_project\\PS_paper\\test\\worldview2\\test_sim\\MS')



n = len(PAN_test_data)
##n=100
ratio=4
params = loadmat('C:/Tools/Python/Python35/RemoteSensingEx/myCodes/training/newdata/200_070964net_noidices_L1_1203.mat')
SAM_v = np.empty(n)
RMSE_v = np.empty(n)
ERGAS_v = np.empty(n)
QAVE_v = np.empty(n)


for i in range(n):
           print (PAN_test_data[i].shape)
           # Pan_data=np.expand_dims(PAN_test_data[i,:,:],axis=0)
           # MS_data = np.squeeze(MS_test_data[i,:,:,:]).transpose(2,0,1)
           Pan_data = np.expand_dims(PAN_test_data[i], axis=0)
           MS_data = np.squeeze(MS_test_data[i,:,:,:]).transpose(2,0,1)
           

###testing

           MS_HR = PNN_test(MS_data,Pan_data,params,mode='reduced')


           
           
           

###visualization
           #plt.close('all')

           #plt.ion()
           #plt.figure()

           #th_PAN = image_quantile(Pan_data,np.array([0.01,0.99]))
           #PAN=image_stretch(np.squeeze(Pan_data),np.squeeze(th_PAN))
           #plt.imshow( image_stretch(np.squeeze(Pan_data),np.squeeze(th_PAN)),cmap='gray',clim=[0,1])
           #plt.title('PANCHROMATIC'), plt.axis('off')


           #RGB_indexes=np.array([3,2,1])
           #RGB_indexes = RGB_indexes - 1


           #plt.figure()
           #th_MSrgb=image_quantile(np.squeeze(MS_data[RGB_indexes,:,:]), np.array([0.01, 0.99]));
           #d=image_stretch(np.squeeze(MS_data[RGB_indexes,:,:]),th_MSrgb)
           #d[d<0]=0
           #d[d>1]=1


           #plt.imshow(d.transpose(1,2,0))
           #plt.title('MULTISPECTRAL LOW RESOLUTION'), plt.axis('off')



           #plt.figure()
           MS_HR = np.squeeze(MS_HR)
           MS_HR = MS_HR.transpose(2,0,1)
           #c=image_stretch(np.squeeze(MS_HR[RGB_indexes,:,:]),th_MSrgb)
           #c[c<0]=0
           #c[c>1]=1
           #plt.imshow(c.transpose(1,2,0))

           #plt.title('Pansharpened'), plt.axis('off')


           I_MS_LR,I_PAN = downgraded(MS_data,Pan_data,ratio)
           

           I_MS_1=scipy.ndimage.interpolation.zoom(I_MS_LR[0,:,:],4,order=3,prefilter=False)
           I_MS_2=scipy.ndimage.interpolation.zoom(I_MS_LR[1,:,:],4,order=3,prefilter=False)
           I_MS_3=scipy.ndimage.interpolation.zoom(I_MS_LR[2,:,:],4,order=3,prefilter=False)
           I_MS_4=scipy.ndimage.interpolation.zoom(I_MS_LR[3,:,:],4,order=3,prefilter=False)

           I_MS_1 = np.expand_dims(I_MS_1,axis=0)
           I_MS_2 = np.expand_dims(I_MS_2,axis=0)
           I_MS_3 = np.expand_dims(I_MS_3,axis=0)
           I_MS_4 = np.expand_dims(I_MS_4,axis=0)
           

           MS_US=np.stack((I_MS_1,I_MS_2,I_MS_3,I_MS_4),axis=0)
           MS_US=np.squeeze(MS_US)
           MS_US=MS_US.transpose(1,2,0)
           MS_HR=MS_HR.transpose(1,2,0)
           MS_data=MS_data.transpose(1,2,0)


           


##### compute the metrics
           #SAM = SAM(MS_HR,MS_US)+SAM
           #RMSE=RMSE(MS_HR,MS_US)+RMSE
           #ERGAS= ERGAS(MS_HR,MS_US)+ERGAS
           #QAVE=QAVE(MS_HR,MS_US)+QAVE

           SAM_v[i] = SAM(MS_HR,MS_US)
           RMSE_v[i]=RMSE(MS_HR,MS_US)
           ERGAS_v[i]= ERGAS(MS_HR,MS_US)
           QAVE_v[i]=QAVE(MS_HR,MS_US)

           print('number',i)

           print('QAVE:%.4f'%QAVE_v[i],'SAM:%.4f'%SAM_v[i],'ERGAS:%.4f'%ERGAS_v[i],'RMSE:%.4f'%RMSE_v[i])


           ##plt.ioff()
           ##plt.show()


##### compute the average metrics
SAM_avg = sum(SAM_v)/n
RMSE_avg=sum(RMSE_v)/n
ERGAS_avg= sum(ERGAS_v)/n
QAVE_avg=sum(QAVE_v)/n

print('QAVE:%.4f'%QAVE_avg,'SAM:%.4f'%SAM_avg,'ERGAS:%.4f'%ERGAS_avg,'RMSE:%.4f'%RMSE_avg)

