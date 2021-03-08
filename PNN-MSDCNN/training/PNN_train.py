import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import scipy.misc as smc
import scipy.ndimage
import matplotlib.pyplot as plt


import os
from skimage.external import tifffile
import datetime

from read_image import read_image_ms,read_image_pan


### download data



PAN_train_data,PAN_test_data=read_image_pan('C:/Users/USER-1/Documents/Peijuan/imageFromCenter/training/PAN',1)
MS_train_data,MS_test_data=read_image_ms('C:/Users/USER-1/Documents/Peijuan/imageFromCenter/training/MS',1)

weightb=loadmat('C:/Tools/MATLAB/mydoc/networks/params/net64.mat')


#### parameters
ratio=4
L=12
mav_value=2**(np.float32(L))


####  getdata for training()

def getdata(data1,data2):
    size = len(data1)
           

    MS_ref=np.empty((size,60,60,4))
    input_data=np.empty((size,60,60,7))
           
    for i in range(0,size):
        X=data1[i,:,:]
        Y=data2[i,:,:,:]
                      
        ###  normalization                       

        X_=X.astype('float32')/mav_value
        Y_=Y.astype('float32')/mav_value

        ###  downsampling
        X_1=scipy.ndimage.interpolation.zoom(X_,(1./4),order=3,prefilter=False)
                     
        Y_1=scipy.ndimage.interpolation.zoom(Y_[:,:,0],(1./4),order=3,prefilter=False)
                      
        Y_2=scipy.ndimage.interpolation.zoom(Y_[:,:,1],(1./4),order=3,prefilter=False)
        Y_3=scipy.ndimage.interpolation.zoom(Y_[:,:,2],(1./4),order=3,prefilter=False)
        Y_4=scipy.ndimage.interpolation.zoom(Y_[:,:,3],(1./4),order=3,prefilter=False)

                      ### compute the indices
        NDVI=(Y_4-Y_1)/(Y_4+Y_1)
        NDWI=-((Y_4-Y_2)/(Y_2+Y_4))
                    
                      ###  upsampling
        Y_11=scipy.ndimage.interpolation.zoom(Y_1,4,order=3,prefilter=False)
        Y_22=scipy.ndimage.interpolation.zoom(Y_2,4,order=3,prefilter=False)
        Y_33=scipy.ndimage.interpolation.zoom(Y_3,4,order=3,prefilter=False)
        Y_44=scipy.ndimage.interpolation.zoom(Y_4,4,order=3,prefilter=False)

        NDVI_=scipy.ndimage.interpolation.zoom(NDVI,4,order=3,prefilter=False)
        NDWI_=scipy.ndimage.interpolation.zoom(NDWI,4,order=3,prefilter=False)

                      
        MS_ref[i,:,:,:]=Y_
        band=np.array([Y_11.T,Y_22.T,Y_33.T,Y_44.T,X_1.T,NDVI_.T,NDWI_.T])
        input_data[i,:,:,:]=band.T.astype('single')
          

    return input_data,MS_ref

#### getdata for the test 

     
### data preprocessing

BATCH_SIZE = 16
LR = 0.001


with tf.name_scope('inputs'):
           
    tf_x=tf.placeholder(tf.float32, [BATCH_SIZE,76,76,7],name='PAN_input')
    I_X=tf.reshape(tf_x,[-1,76,76,7])

    tf_y=tf.placeholder(tf.float32,[BATCH_SIZE,60,60,4],name='MS_input')

####  CNN

with tf.name_scope('weights'):

    weights = {
        'w1': tf.Variable(weightb['w1'], name='w1'),
        'w2': tf.Variable(weightb['w2'], name='w2'),
        'w3': tf.Variable(weightb['w3'], name='w3')
            }
           
with tf.name_scope('biases'):
           
    biases = {
        'b1': tf.Variable(weightb['b1'].reshape(64), name='b1'),
        'b2': tf.Variable(weightb['b2'].reshape(32), name='b2'),
        'b3': tf.Variable(weightb['b3'].reshape(4), name='b3')
                    }   


###  conv layer one
with tf.name_scope('conv1'):

    conv1=tf.nn.relu(tf.nn.conv2d(I_X,weights['w1'],strides=[1,1,1,1],padding='VALID')+biases['b1'],name='conv1')
           
    tf.summary.histogram('conv1+/weights', weights['w1'])
    tf.summary.histogram('conv1+/biases', biases['b1'])

###  batch normalization layer
###  conv1_=tf.layers.batch_normalization(conv1,momentum=0.4)

###  conv layer two
with tf.name_scope('conv2'):
    conv2=tf.nn.relu(tf.nn.conv2d(conv1,weights['w2'],strides=[1,1,1,1],padding='VALID')+biases['b2'],name='conv1')
    tf.summary.histogram('conv2+/weights', weights['w2'])
    tf.summary.histogram('conv2+/biases', biases['b2'])
           
###  conv layer three
with tf.name_scope('conv3'):
    conv3=tf.nn.conv2d(conv2,weights['w3'],strides=[1,1,1,1],padding='VALID')+biases['b3']
    tf.summary.histogram('conv3+/weights', weights['w3'])
    tf.summary.histogram('conv3+/biases', biases['b3'])
    output = conv3
    tf.summary.histogram('conv3+/outputs', output)
           

### compute the loss function & optimizer
###loss=tf.losses.mean_squared_error(tf_y,output,weights=1.0,scope=None)

with tf.name_scope('loss'):
##  loss= tf.reduce_mean(tf.square(tf_y-output))
##  L1 loss 
    loss =tf.reduce_mean( tf.abs(tf_y - output))     
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    optimizer=tf.train.GradientDescentOptimizer(LR)
    train_op=optimizer.minimize(loss=loss)


###  get all the parameters of the training
params=tf.trainable_variables()

print("trainable variables-----")
for idx,v in enumerate(params):
    print("param{:3}:{:15}",idx,v.name)

                     
### create the saver
saver = tf.train.Saver()


### create session
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)

           ### prepare the inputdata 
           #input_data,MS_ref = getdata(PAN_train_data,MS_train_data)
           #input_data = np.pad(input_data, ((0,0),(8,8),(8,8),(0,0)),mode='edge')
           #print('input_data:',input_data.shape)
           

           
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("C:/Tools/Python/logs", tf.get_default_graph())
           
           
    for step in range(100):
        starttime = datetime.datetime.now()

        batch_idx= len(PAN_train_data)//BATCH_SIZE
                      
        for idx in range(0,batch_idx):
            pan_train = PAN_train_data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            ms_train = MS_train_data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            input_data,MS_ref = getdata(pan_train,ms_train)
            input_data = np.pad(input_data, ((0,0),(8,8),(8,8),(0,0)),mode='edge')

                                 
            batch_imgs = input_data
            batch_labels = MS_ref
            _, loss_ = sess.run([train_op, loss], {tf_x: batch_imgs, tf_y: batch_labels})


        endtime = datetime.datetime.now()
        time = (endtime-starttime).seconds
                 
        if step % 1 == 0:
            print('Step:',step,'| train loss: %.6f' % loss_,'| %.2f seconds/step' % time)

            result = sess.run(merged,feed_dict={tf_x: batch_imgs, tf_y: batch_labels})
            writer.add_summary(result, step)
        
        if (step+1) % 50 == 0:
            save_path=saver.save(sess,"C:/Tools/Python/Python35/RemoteSensingEx/myCodes/training/newdata/parameter/PANCNN64net.ckpt")

            W1=sess.run(params[0])
            W2=sess.run(params[1])
            W3=sess.run(params[2])
            b1=sess.run(params[3])
            b2=sess.run(params[4])
            b3=sess.run(params[5])

            savemat('%s' % (step+1)+'_070964net_0304.mat',{'W1':W1,'W2':W2,'W3':W3,'b1':b1,'b2':b2,'b3':b3})
           


