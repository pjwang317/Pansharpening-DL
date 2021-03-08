
import numpy as np

from scipy.io import loadmat
import matplotlib.pyplot as plt
from skimage import io



test_MS=loadmat('C:/Tools/MATLAB/PANCNN/testMS.mat')
I_MS=test_MS['PMS_100']
print(I_MS.shape)



def image_quantile(img,p):
    print(img.shape)
    Nk, Nr, Nc = img.shape
    print(Nk, Nr, Nc )
    y = np.zeros((Nk, np.size(p)))
    for index in range(Nk):
        y[index,:] = quantile( img[index,:,:], p )
    return y

def quantile(x, p):
    print(x)
    x = np.sort(x.flatten())
    print(x)
    p = np.maximum(np.floor(np.dot(np.size(x),p)), 0).astype('int')
    y=x[p]
    return y
    
def image_stretch(img, th):
    img = np.double(img)
    if np.size(th)==2:
        img = (img-th[0])/(th[1]-th[0])
    else:
        Nk,Nr,Nc = img.shape
        for index in range(Nk):
            img[index,:,:] = (img[index,:,:]-th[index,0])/(th[index,1]-th[index,0])
        
    return img





band1=I_MS[:,:,0]
band2=I_MS[:,:,1]
band3=I_MS[:,:,2]
print(band1.shape)
print(band2.shape)
print(band3.shape)



I_MS1=np.array([band3,band2,band1])
print(I_MS1.shape)

th_MS=image_quantile(I_MS1,[0.01,0.99])
img=image_stretch(I_MS1,th_MS)



band1=img[0,:,:]
band2=img[1,:,:]
band3=img[2,:,:]
print(band1.shape)
print(band2.shape)
print(band3.shape)
print(band1)
print(band2)
print(band3)


img_=np.array([band1.T,band2.T,band3.T])
print(img_.shape)


img_=img_.T
print(img_.shape)
img_=np.floor(np.dot(img_,255)).astype('int')
print(img_)



plt.imshow(img_)
plt.axis('off')
plt.show()



