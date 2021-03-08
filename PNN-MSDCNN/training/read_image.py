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
           
    MS_train_data = numpy.empty([train_size,60,60,4])
    MS_test_data = numpy.empty([test_size,60,60,4])

    while train_size>0:
        imag=tifffile.imread(MS_train[train_size-1])
        shape_array = imag.shape
        std_array=tuple(numpy.array([60,60,4]))
        if (shape_array==std_array)==True:
            MS_train_data[train_size-1,:,:,:]=imag.reshape([-1,60,60,4])
        else:
            os.remove(MS_train[train_size-1])
        train_size=train_size-1         

    while test_size>0:
        imag=tifffile.imread(MS_test[test_size-1])
        shape_array = imag.shape
        std_array=tuple(numpy.array([60,60,4]))
        if (shape_array==std_array)==True:
            MS_test_data[test_size-1,:,:,:]=imag.reshape([-1,60,60,4])
        else:
            os.remove(MS_test[test_size-1])
        test_size=test_size-1
    return MS_train_data,MS_test_data
           
def read_image_pan(path,train_ratio):
    img = get_list(path)
    d=len(img)

    train_size = int(d*train_ratio)
    test_size=d-train_size
    random.seed(0)
    random.shuffle(img)
           
           

    PAN_train = img[:train_size]
    PAN_test =  img[train_size:]
           #print(PAN_test[0])
           
    PAN_train_data = numpy.empty([train_size,240,240])
    PAN_test_data = numpy.empty([test_size,240,240])


    while train_size>0:
        imag=tifffile.imread(PAN_train[train_size-1])
                      
        shape_array = imag.shape
        
        std_array=tuple(numpy.array([240,240]))
           
        if (shape_array==std_array)==True:
            PAN_train_data[train_size-1,:,:]=imag.reshape([-1,240,240])
                     
        else:
            os.remove(PAN_train[train_size-1])
           
        train_size=train_size-1         

    while test_size>0:
        imag=tifffile.imread(PAN_test[test_size-1])
                      
        shape_array = imag.shape
        
        std_array=tuple(numpy.array([240,240]))
           
        if (shape_array==std_array)==True:
            PAN_test_data[test_size-1,:,:]=imag.reshape([-1,240,240])
                     
        else:
            os.remove(PAN_test[test_size-1])
           
        test_size=test_size-1
                      
    return PAN_train_data,PAN_test_data


      



