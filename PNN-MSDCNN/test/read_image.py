import os
import numpy
import random
from skimage.external import tifffile



def get_list(path):
           return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.tif')]

def read_image_ms(path,train_ratio):
           img = get_list(path)
           d=len(img)
           
           train_size = int(d*train_ratio)
           test_size=d-train_size
           random.seed(0)
           random.shuffle(img)

           
           
           MS_train = img[:train_size]
           MS_test = img[train_size:]
           for i in range(test_size):
                      print('number:',i)
                      print(MS_test[i])




           
           MS_train_data = numpy.empty([train_size,200,200,4])
           MS_test_data = numpy.empty([test_size,200,200,4])


           while train_size>0:
                      imag=tifffile.imread(MS_train[train_size-1])
                      
                      shape_array = imag.shape
        
                      std_array=tuple(numpy.array([200,200,4]))
           
                      if (shape_array==std_array)==True:
                                 MS_train_data[train_size-1,:,:,:]=imag.reshape([-1,200,200,4])
                     
                      else:
                                 os.remove(MS_train[train_size-1])
           
                      train_size=train_size-1         

           while test_size>0:
                      imag=tifffile.imread(MS_test[test_size-1])
                      
                      shape_array = imag.shape
        
                      std_array=tuple(numpy.array([200,200,4]))
           
                      if (shape_array==std_array)==True:
                                 MS_test_data[test_size-1,:,:,:]=imag.reshape([-1,200,200,4])
                    
                      else:
                                 os.remove(MS_test[test_size-1])
           
                      test_size=test_size-1
           return MS_train_data,MS_test_data
           
def read_image_pan(path,train_ratio):
           img = get_list(path)
           d=len(img)
           print('d',d)

           train_size = int(d*train_ratio)
           test_size=d-train_size
           random.seed(0)
           random.shuffle(img)
           
           

           PAN_train = img[:train_size]
           PAN_test =  img[train_size:]
           for i in range(test_size):
                      print('number:',i)
                      print(PAN_test[i])
           
           PAN_train_data = numpy.empty([train_size,800,800])
           PAN_test_data = numpy.empty([test_size,800,800])


           while train_size>0:
                      imag=tifffile.imread(PAN_train[train_size-1])
                      
                      shape_array = imag.shape
        
                      std_array=tuple(numpy.array([800,800]))
           
                      if (shape_array==std_array)==True:
                                 PAN_train_data[train_size-1,:,:]=imag.reshape([-1,800,800])
                     
                      else:
                                 os.remove(PAN_train[train_size-1])
           
                      train_size=train_size-1         

           while test_size>0:
                      imag=tifffile.imread(PAN_test[test_size-1])
                      
                      shape_array = imag.shape
        
                      std_array=tuple(numpy.array([800,800]))
           
                      if (shape_array==std_array)==True:
                                 PAN_test_data[test_size-1,:,:]=imag.reshape([-1,800,800])
                     
                      else:
                                 os.remove(PAN_test[test_size-1])
           
                      test_size=test_size-1
                      
           return PAN_train_data,PAN_test_data


      



