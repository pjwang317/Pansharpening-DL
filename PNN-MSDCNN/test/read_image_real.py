import os
import numpy
import random
from skimage.external import tifffile



def get_list(path):
           return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.tif')]

def read_image_ms(path):
           img = get_list(path)
           # print(img)
           d=len(img)
           
           #train_size = int(d*train_ratio)
           #test_size=d-train_size
           #random.seed(0)
           #random.shuffle(img)

           
           
           #MS_train = img[:train_size]
           #MS_test = img[train_size:]
           #for i in range(test_size):
           #           print('number:',i)
           #           print(MS_test[i])




           
           MS_test_data = numpy.empty([d,200,200,4])     

           while d>0:
                      imag=tifffile.imread(img[d-1])
                      
                      shape_array = imag.shape
        
                      std_array=tuple(numpy.array([200,200,4]))
           
                      if (shape_array==std_array)==True:
                                 MS_test_data[d-1,:,:,:]=imag.reshape([-1,200,200,4])
                    
                      else:
                                 os.remove(img[d-1])
           
                      d=d-1
           
           return MS_test_data,img
           
def read_image_pan(path):
           img = get_list(path)
           d=len(img)
         

           #train_size = int(d*train_ratio)
           #test_size=d-train_size
           #random.seed(0)
           #random.shuffle(img)
           
           

           #PAN_train = img[:train_size]
           #PAN_test =  img[train_size:]
           #for i in range(test_size):
           #          print('number:',i)
           #          print(PAN_test[i])
           
           #PAN_train_data = numpy.empty([train_size,800,800])
           PAN_test_data = numpy.empty([d,800,800])


           

           while d>0:
                      imag=tifffile.imread(img[d-1])
                      
                      shape_array = imag.shape
        
                      std_array=tuple(numpy.array([800,800]))
           
                      if (shape_array==std_array)==True:
                                 PAN_test_data[d-1,:,:]=imag.reshape([-1,800,800])
                     
                      else:
                                 os.remove(img[d-1])
           
                      d=d-1
           #print(PAN_test)           
           return PAN_test_data,img


      



